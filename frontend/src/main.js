/**
 * SPA Router & Application Entry Point
 * UAV Wildfire Early Detection System
 */

import "./style.css";
import { createNavbar, updateNavActive } from "./components/navbar.js";
import { renderLanding } from "./pages/landing.js";
import { renderDetect } from "./pages/detect.js";

/* ============================================
   ROUTER
   ============================================ */
const routes = {
  home: renderLanding,
  detect: renderDetect,
};

let currentPage = null;

function navigate(page) {
  if (currentPage === page) return;
  currentPage = page;

  const appContainer = document.getElementById("app");
  const renderFn = routes[page] || routes.home;

  // Clear existing lightbox if any
  const existingLightbox = document.getElementById("gallery-lightbox");
  if (existingLightbox) existingLightbox.remove();

  renderFn(appContainer);
  updateNavActive(page);

  // Scroll to top on page change
  window.scrollTo({ top: 0, behavior: "instant" });
}

function getPageFromHash() {
  const hash = window.location.hash.replace("#", "").split("?")[0];
  return hash || "home";
}

/* ============================================
   INITIALIZATION
   ============================================ */
function init() {
  // Insert navbar
  const navbar = createNavbar();
  document.body.prepend(navbar);

  // Create app container
  let appContainer = document.getElementById("app");
  if (!appContainer) {
    appContainer = document.createElement("main");
    appContainer.id = "app";
    appContainer.className = "pt-[72px]"; // Offset for fixed navbar
    document.body.appendChild(appContainer);
  }

  // Listen for hash changes
  window.addEventListener("hashchange", () => {
    navigate(getPageFromHash());
  });

  // Initial render
  navigate(getPageFromHash());
}

// Boot
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
