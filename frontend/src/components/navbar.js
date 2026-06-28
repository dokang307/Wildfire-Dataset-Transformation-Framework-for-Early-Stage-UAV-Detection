/**
 * Navigation bar component.
 * Handles active state tracking and smooth scroll.
 */

export function createNavbar() {
  const nav = document.createElement("nav");
  nav.id = "main-navbar";
  nav.className =
    "fixed top-0 left-0 right-0 z-50 transition-all duration-300";
  nav.innerHTML = `
    <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
      <!-- Logo -->
      <a href="#home" class="flex items-center gap-3 group" id="nav-logo">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-accent to-accent-light flex items-center justify-center shadow-lg group-hover:shadow-accent-glow transition-shadow duration-300">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10"/>
            <path d="M12 8c-2 0-4 2-4 4s2 4 4 4 4-2 4-4"/>
            <circle cx="12" cy="12" r="1.5"/>
          </svg>
        </div>
        <span class="font-display font-bold text-xl text-text-primary group-hover:text-ember transition-colors">
          UAV<span class="text-accent-light">Fire</span>Detect
        </span>
      </a>

      <!-- Nav Links -->
      <div class="hidden md:flex items-center gap-1">
        <a href="#home" class="nav-link active" data-page="home">
          <span>Home</span>
        </a>
        <a href="#detect" class="nav-link" data-page="detect">
          <span>Detection</span>
        </a>
      </div>

      <!-- Mobile menu button -->
      <button id="mobile-menu-btn" class="md:hidden w-10 h-10 rounded-lg bg-bg-tertiary flex items-center justify-center text-text-primary hover:bg-bg-card-hover transition-colors">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="3" y1="6" x2="21" y2="6"/>
          <line x1="3" y1="12" x2="21" y2="12"/>
          <line x1="3" y1="18" x2="21" y2="18"/>
        </svg>
      </button>
    </div>

    <!-- Mobile menu -->
    <div id="mobile-menu" class="md:hidden hidden px-6 pb-4">
      <a href="#home" class="mobile-nav-link" data-page="home">Home</a>
      <a href="#detect" class="mobile-nav-link" data-page="detect">Detection</a>
    </div>
  `;

  // Style nav links
  const style = document.createElement("style");
  style.textContent = `
    .nav-link {
      padding: 8px 20px;
      border-radius: 10px;
      font-weight: 500;
      font-size: 0.95rem;
      color: var(--color-text-secondary);
      text-decoration: none;
      transition: all 0.25s ease;
      position: relative;
    }
    .nav-link:hover {
      color: var(--color-text-primary);
      background: var(--color-bg-tertiary);
    }
    .nav-link.active {
      color: var(--color-ember);
      background: rgba(139, 26, 26, 0.15);
    }
    .mobile-nav-link {
      display: block;
      padding: 12px 16px;
      border-radius: 10px;
      font-weight: 500;
      color: var(--color-text-secondary);
      text-decoration: none;
      transition: all 0.2s;
    }
    .mobile-nav-link:hover, .mobile-nav-link.active {
      color: var(--color-ember);
      background: rgba(139, 26, 26, 0.15);
    }
    #main-navbar {
      background: rgba(10, 10, 10, 0.85);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border-bottom: 1px solid transparent;
    }
    #main-navbar.scrolled {
      border-bottom-color: var(--color-border);
    }
  `;
  document.head.appendChild(style);

  // Mobile menu toggle
  nav.querySelector("#mobile-menu-btn").addEventListener("click", () => {
    const menu = nav.querySelector("#mobile-menu");
    menu.classList.toggle("hidden");
  });

  // Scroll effect
  window.addEventListener("scroll", () => {
    nav.classList.toggle("scrolled", window.scrollY > 20);
  });

  return nav;
}

/**
 * Update active nav link based on current page.
 */
export function updateNavActive(page) {
  document.querySelectorAll(".nav-link, .mobile-nav-link").forEach((link) => {
    const linkPage = link.getAttribute("data-page");
    link.classList.toggle("active", linkPage === page);
  });
}
