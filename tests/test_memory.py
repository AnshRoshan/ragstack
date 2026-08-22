"""Conversation memory, query rewriting, strip refinement, and session flows."""

from __future__ import annotations

from conftest import FakeEmbeddings, FakeLLM

from ragstack.memory import ConversationMemory, format_history, rewrite_question
from ragstack.retrieval.evaluator import EvidenceGrader
from ragstack.service import RAGStack
from ragstack.types import RetrievedItem


class TestConversationMemory:
    def test_append_history_order(self, tmp_path):
        mem = ConversationMemory(tmp_path / "mem")
        mem.append("s1", "user", "what is X?")
        mem.append("s1", "assistant", "X is a thing.")
        mem.append("s1", "user", "why does it matter?")
        hist = mem.history("s1")
        assert [h["content"] for h in hist] == ["what is X?", "X is a thing.", "why does it matter?"]
        assert hist[-1]["role"] == "user"

    def test_limit_and_session_isolation(self, tmp_path):
        mem = ConversationMemory(tmp_path / "mem")
        for i in range(6):
            mem.append("s1", "user", f"msg {i}")
            mem.append("s1", "assistant", f"reply {i}")
        mem.append("s2", "user", "other thread")
        assert len(mem.history("s1", limit=4)) == 4
        assert mem.history("s2")[0]["content"] == "other thread"
        stats = mem.stats()
        assert stats["sessions"] == 2 and stats["turns"] == 13

    def test_clear(self, tmp_path):
        mem = ConversationMemory(tmp_path / "mem")
        mem.append("a", "user", "hi")
        mem.append("b", "user", "hi")
        mem.clear("a")
        assert mem.history("a") == [] and mem.history("b")


class TestRewrite:
    def test_rewrites_followup_with_history(self):
        llm = FakeLLM([{"content": "What is the graphics card of the ASUS ROG G15?"}])
        history = [
            {"role": "user", "content": "I'm looking for a gaming laptop"},
            {"role": "assistant", "content": "I recommend the ASUS ROG G15."},
        ]
        out = rewrite_question(llm, "and the graphics card?", history)
        assert "ASUS" in out

    def test_no_history_returns_original(self):
        llm = FakeLLM([{"content": "SHOULD NOT BE CALLED"}])
        assert rewrite_question(llm, "plain question", []) == "plain question"
        assert llm.calls == []

    def test_llm_failure_falls_back(self):
        class Boom(FakeLLM):
            def chat(self, *a, **k):
                raise RuntimeError("down")

        out = rewrite_question(Boom([]), "original question?", [{"role": "user", "content": "h"}])
        assert out == "original question?"

    def test_format_history_truncates(self):
        big = [{"role": "user", "content": "x" * 3000}]
        assert len(format_history(big)) <= 2400


class _OverlapReranker:
    """Fake cross-encoder: scores strips by keyword overlap with the query."""

    def rerank(self, query, items, text_key="text"):
        q = set(query.lower().split())
        return sorted(items, key=lambda s: len(q & set(getattr(s, text_key).lower().split())), reverse=True)


class TestStripRefinement:
    def test_refine_keeps_relevant_sentences(self):
        g = EvidenceGrader(reranker=_OverlapReranker())
        item = RetrievedItem(
            chunk_id="c1",
            source="s.md",
            title="T",
            text=(
                "Kubernetes schedules pods across worker nodes automatically. "
                "The chef recommends boiling pasta water with salt. "
                "Rolling updates keep services online during deploys. "
                "A sonnet is a fourteen line poem with strict rhyme. "
                "Horizontal scaling reacts to cluster load within minutes."
            ),
        )
        refined = g.refine("kubernetes pods scaling updates", [item], max_strips=3)
        assert len(refined) == 1
        joined = refined[0].text.lower()
        assert "kubernetes" in joined or "scaling" in joined or "updates" in joined
        assert "pasta" not in joined and "sonnet" not in joined

    def test_refine_without_crossencoder_is_identity(self):
        g = EvidenceGrader()  # no reranker
        items = [RetrievedItem(chunk_id="c", source="s", title="t", text="long enough text here " * 5)]
        assert g.refine("q", items) is items


class TestSessionFlow:
    def _svc(self, app_config):
        svc = RAGStack(app_config)
        svc._embeddings = FakeEmbeddings()
        return svc

    def test_first_turn_no_rewrite_second_turn_rewrites(self, app_config, tmp_path):
        svc = self._svc(app_config)
        doc = tmp_path / "d.md"
        doc.write_text("Content about vector databases and LanceDB. " * 30, encoding="utf-8")
        svc.ingest([doc], with_graph=False)
        svc._llm = FakeLLM([{"content": "Answer one."}, {"content": "standalone rewritten question"}, {"content": "Answer two."}])

        first_events = list(svc.stream_query("what stores vectors?", mode="vector", use_cache=False, session_id="s"))
        assert not any(e["type"] == "rewrite" for e in first_events)

        second_events = list(svc.stream_query("and who made it?", mode="vector", use_cache=False, session_id="s"))
        rewrites = [e for e in second_events if e["type"] == "rewrite"]
        assert rewrites and rewrites[0]["standalone"] == "standalone rewritten question"
        # rewrite call + synthesis call = 2 LLM calls this turn; history carried through
        assert len(svc._llm.calls) == 3
        mem_stats = svc.status()["memory"]
        assert mem_stats["turns"] == 4  # 2 turns stored per answered question

    def test_agent_mode_history_in_messages(self, app_config, tmp_path):
        svc = self._svc(app_config)
        doc = tmp_path / "d.md"
        doc.write_text("Some indexed content. " * 40, encoding="utf-8")
        svc.ingest([doc], with_graph=False)
        svc._llm = FakeLLM(
            [
                # turn 1 (no history yet): agent asks for context, then answers
                {"tool_calls": [{"name": "search_chunks", "args": {"query": "indexed content"}}]},
                {"content": "Done [S1]."},
                # turn 2: rewrite, then agent synthesis — this call must carry history
                {"content": "standalone follow-up"},
                {"content": "Answer two."},
            ]
        )
        list(svc.stream_query("first question?", mode="auto", use_cache=False, session_id="s"))
        second = list(svc.stream_query("follow-up?", mode="auto", use_cache=False, session_id="s"))
        assert any(e["type"] == "rewrite" and e["standalone"] == "standalone follow-up" for e in second)
        synthesis_msgs = svc._llm.calls[-1]
        roles = [m["role"] for m in synthesis_msgs]
        assert roles.count("assistant") >= 1  # prior assistant turn present as history
        assert any(m["role"] == "user" and "first question?" in m["content"] for m in synthesis_msgs)

    def test_memory_disabled(self, app_config, tmp_path):
        app_config.agent.memory_turns = 0
        svc = self._svc(app_config)
        svc._llm = FakeLLM([{"content": "ans"}])
        list(svc.stream_query("q?", mode="vector", use_cache=False, session_id="s"))
        assert svc.memory is None
        assert svc.status()["memory"]["turns"] == 0

    def test_forget_clears_session(self, app_config):
        svc = self._svc(app_config)
        svc.memory.append("s", "user", "hello")
        svc.memory.clear("s")
        assert svc.memory.history("s") == []
