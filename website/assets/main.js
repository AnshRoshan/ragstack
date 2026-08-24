/* Shared site behaviour: theme + reveal. Page-specific JS stays inline per page. */
(function () {
  const root = document.documentElement;
  const stored = localStorage.getItem('ragstack-theme');
  const prefers = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  root.setAttribute('data-theme', stored || prefers);

  function refreshToggle() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.setAttribute('aria-label',
      root.getAttribute('data-theme') === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }

  document.addEventListener('DOMContentLoaded', () => {
    refreshToggle();
    const io = new IntersectionObserver((es) => es.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
    }), { threshold: 0.12 });
    document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  });

  window.ragstackSetTheme = function (theme) {
    root.setAttribute('data-theme', theme);
    localStorage.setItem('ragstack-theme', theme);
    const m = document.querySelector('meta[name="theme-color"]');
    if (m) m.content = getComputedStyle(document.documentElement).getPropertyValue('--paper').trim();
    refreshToggle();
    document.dispatchEvent(new CustomEvent('ragstack:theme', { detail: { theme } }));
  };

  document.addEventListener('click', (e) => {
    if (e.target.closest('#themeToggle')) {
      const next = root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      window.ragstackSetTheme(next);
    }
  });
})();
