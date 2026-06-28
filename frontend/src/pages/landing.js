/**
 * Landing page — Hero, Abstract, Metrics, Gallery, Timeline.
 */

import { createTimeline } from "../components/timeline.js";

/* ============================================
   METRICS DATA
   ============================================ */
const METRICS = [
  { label: "Precision", value: 98.68, suffix: "%", icon: "🎯" },
  { label: "Recall", value: 96.16, suffix: "%", icon: "📡" },
  { label: "mAP@50", value: 97.97, suffix: "%", icon: "📊" },
  { label: "F1 Score", value: 0.97, suffix: "", icon: "⚡" },
];

/* ============================================
   GALLERY IMAGES
   ============================================ */
const GALLERY_IMAGES = [
  { src: "/figures/results.png", label: "Training Results" },
  { src: "/figures/BoxPR_curve.png", label: "Precision-Recall Curve" },
  { src: "/figures/BoxF1_curve.png", label: "F1-Confidence Curve" },
  { src: "/figures/confusion_matrix.png", label: "Confusion Matrix" },
  {
    src: "/figures/confusion_matrix_normalized.png",
    label: "Normalized Confusion Matrix",
  },
  { src: "/figures/labels.jpg", label: "Label Distribution" },
  { src: "/figures/val_batch0_pred.jpg", label: "Validation Predictions 1" },
  { src: "/figures/val_batch1_pred.jpg", label: "Validation Predictions 2" },
  { src: "/figures/val_batch2_pred.jpg", label: "Validation Predictions 3" },
];

/* ============================================
   RENDER LANDING PAGE
   ============================================ */
export function renderLanding(container) {
  container.innerHTML = "";
  container.className = "page-container page-enter";

  // === HERO SECTION ===
  const hero = document.createElement("section");
  hero.className =
    "relative min-h-[85vh] flex items-center justify-center overflow-hidden";
  hero.innerHTML = `
    <!-- Video Background -->
    <div class="absolute inset-0 z-0">
      <video src="/wflol.mp4" autoplay loop muted playsinline class="w-full h-full object-cover opacity-50"></video>
      <!-- Gradient overlay for readability -->
      <div class="absolute inset-0 bg-gradient-to-t from-bg-primary via-bg-primary/80 to-transparent"></div>
    </div>

    <!-- Content -->
    <div class="relative z-10 text-center px-6 max-w-4xl mx-auto">
      <div class="animate-fade-in-up">
        <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 border border-accent/30 mb-8">
          <span class="w-2 h-2 rounded-full bg-ember animate-pulse"></span>
          <span class="text-sm font-medium text-accent-light">YOLOv11n • ONNX Optimized</span>
        </div>
      </div>

      <h1 class="font-display font-900 text-5xl md:text-7xl lg:text-8xl leading-tight mb-6 animate-fade-in-up-delay-1">
        <span class="text-text-primary">UAV Wildfire</span><br />
        <span class="gradient-text">Early Detection</span>
      </h1>

      <p class="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto mb-10 leading-relaxed animate-fade-in-up-delay-2">
        Real-time detection of early fire and smoke from UAV aerial imagery
        using deep learning. Achieving <span class="text-ember font-semibold">97.97% mAP@50</span> with
        optimized ONNX inference.
      </p>

      <div class="flex flex-col sm:flex-row gap-4 justify-center animate-fade-in-up-delay-3">
        <a href="#detect" class="btn-accent text-lg px-8 py-4 inline-flex items-center gap-2">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
          </svg>
          Try Detection
        </a>
        <a href="#timeline-section" class="px-8 py-4 rounded-xl border border-border text-text-secondary hover:text-text-primary hover:border-accent/50 transition-all duration-300 inline-flex items-center gap-2 text-lg font-medium">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          View Timeline
        </a>
      </div>
    </div>

    <!-- Scroll indicator -->
    <div class="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-text-muted">
        <polyline points="6 9 12 15 18 9"/>
      </svg>
    </div>
  `;

  container.appendChild(hero);



  // === ABSTRACT SECTION ===
  const abstract = document.createElement("section");
  abstract.className = "py-20 px-6";
  abstract.innerHTML = `
    <div class="max-w-4xl mx-auto">
      <h2 class="section-title text-center gradient-text">About This Project</h2>
      <p class="section-subtitle text-center">Abstract</p>

      <div class="glass-card p-8 md:p-10">
        <p class="text-text-secondary leading-relaxed text-lg mb-6">
          Wildfires pose an increasing threat to ecosystems, communities, and infrastructure worldwide.
          Early detection is critical to minimizing damage and enabling rapid response. This project
          presents a <span class="text-text-primary font-semibold">deep learning–based wildfire early detection system</span>
          designed to operate on imagery captured by <span class="text-text-primary font-semibold">Unmanned Aerial Vehicles (UAVs)</span>.
        </p>
        <p class="text-text-secondary leading-relaxed text-lg mb-6">
          We leverage the <span class="text-ember font-semibold">YOLOv11n</span> (nano) architecture — a state-of-the-art
          real-time object detection model — fine-tuned to identify two critical early indicators of wildfire:
        </p>
        <div class="grid sm:grid-cols-2 gap-4 mb-6">
          <div class="flex items-center gap-4 p-4 rounded-xl bg-bg-tertiary border border-border">
            <span class="text-3xl">🔴</span>
            <div>
              <p class="font-semibold text-text-primary">Early Fire</p>
              <p class="text-sm text-text-muted">Nascent flame regions visible from aerial perspectives</p>
            </div>
          </div>
          <div class="flex items-center gap-4 p-4 rounded-xl bg-bg-tertiary border border-border">
            <span class="text-3xl">🟠</span>
            <div>
              <p class="font-semibold text-text-primary">Early Smoke</p>
              <p class="text-sm text-text-muted">Smoke plumes indicative of fire ignition</p>
            </div>
          </div>
        </div>
        <p class="text-text-secondary leading-relaxed text-lg">
          The model is trained on a curated dataset of <span class="text-text-primary font-semibold">20,939 annotated instances</span>
          and optimized for deployment using <span class="text-text-primary font-semibold">ONNX Runtime</span>,
          enabling efficient inference on cloud platforms and edge devices.
        </p>
      </div>
    </div>
  `;
  container.appendChild(abstract);

  // === METRICS SECTION ===
  const metrics = document.createElement("section");
  metrics.className = "relative py-24 px-6 overflow-hidden bg-bg-primary";
  metrics.innerHTML = `
    <!-- Background gradient & Embers (Moved from Hero) -->
    <div class="absolute inset-0 bg-gradient-to-br from-bg-primary via-[#120808] to-bg-primary z-0"></div>
    <div class="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(139,26,26,0.15),transparent_60%)] z-0"></div>
    <div class="absolute inset-0 bg-[radial-gradient(ellipse_at_70%_80%,rgba(178,34,34,0.08),transparent_50%)] z-0"></div>
    <div class="ember-container" id="metrics-ember-container"></div>
    
    <div class="relative z-10 max-w-5xl mx-auto">
      <h2 class="section-title text-center gradient-text">Performance Metrics</h2>
      <p class="section-subtitle text-center">Best checkpoint results on validation set</p>

      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6" id="metrics-grid">
        ${METRICS.map(
          (m) => `
          <div class="glass-card p-6 text-center group hover:accent-glow">
            <span class="text-3xl mb-3 block">${m.icon}</span>
            <div class="font-display font-bold text-3xl md:text-4xl text-text-primary mb-1 metric-value"
                 data-target="${m.value}" data-suffix="${m.suffix}">
              0${m.suffix}
            </div>
            <p class="text-sm text-text-muted font-medium">${m.label}</p>
          </div>
        `
        ).join("")}
      </div>
    </div>
  `;
  container.appendChild(metrics);

  // Animate metrics on scroll and spawn embers
  observeMetrics(metrics);
  spawnEmbers(metrics.querySelector("#metrics-ember-container"));

  // === GALLERY SECTION ===
  const gallery = document.createElement("section");
  gallery.className = "py-20 px-6";
  gallery.innerHTML = `
    <div class="max-w-6xl mx-auto">
      <h2 class="section-title text-center gradient-text">Training Results</h2>
      <p class="section-subtitle text-center">Detailed performance analysis and validation</p>

      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4" id="gallery-grid">
        ${GALLERY_IMAGES.map(
          (img) => `
          <div class="glass-card overflow-hidden cursor-pointer group gallery-item" data-src="${img.src}">
            <div class="aspect-video bg-bg-tertiary overflow-hidden">
              <img src="${img.src}" alt="${img.label}" loading="lazy"
                   class="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" />
            </div>
            <div class="p-3 flex items-center justify-between">
              <span class="text-sm text-text-secondary font-medium">${img.label}</span>
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
                   class="text-text-muted group-hover:text-accent-light transition-colors">
                <polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/>
              </svg>
            </div>
          </div>
        `
        ).join("")}
      </div>
    </div>
  `;
  container.appendChild(gallery);

  // Lightbox for gallery
  setupLightbox(gallery);

  // === TIMELINE SECTION ===
  const timeline = createTimeline();
  container.appendChild(timeline);

  // === FOOTER ===
  const footer = document.createElement("footer");
  footer.className = "py-12 px-6 border-t border-border bg-bg-secondary";
  footer.innerHTML = `
    <div class="max-w-5xl mx-auto text-center">
      <p class="text-text-muted text-sm">
        UAV Wildfire Early Detection System • Built with YOLOv11 + ONNX Runtime
      </p>
      <p class="text-text-muted text-xs mt-2">
        © 2026 DSP-UAV Project
      </p>
    </div>
  `;
  container.appendChild(footer);
}

/* ============================================
   HELPER: Spawn ember particles
   ============================================ */
function spawnEmbers(container) {
  if (!container) return;
  const count = 15;
  for (let i = 0; i < count; i++) {
    const ember = document.createElement("div");
    ember.className = "ember";
    const x = Math.random() * 100;
    const delay = Math.random() * 8;
    const duration = 6 + Math.random() * 8;
    const size = 2 + Math.random() * 4;
    ember.style.left = `${x}%`;
    ember.style.width = `${size}px`;
    ember.style.height = `${size}px`;
    ember.style.animationName =
      Math.random() > 0.5 ? "ember-float" : "ember-float-alt";
    ember.style.animationDuration = `${duration}s`;
    ember.style.animationDelay = `${delay}s`;
    ember.style.animationIterationCount = "infinite";
    ember.style.animationTimingFunction = "ease-out";
    container.appendChild(ember);
  }
}

/* ============================================
   HELPER: Animate metric counters on scroll
   ============================================ */
function observeMetrics(section) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          animateCounters(section);
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.3 }
  );
  observer.observe(section);
}

function animateCounters(section) {
  const els = section.querySelectorAll(".metric-value");
  els.forEach((el) => {
    const target = parseFloat(el.dataset.target);
    const suffix = el.dataset.suffix;
    const duration = 1500;
    const start = performance.now();
    const isFloat = target < 1;

    function update(now) {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = target * eased;

      if (isFloat) {
        el.textContent = current.toFixed(2) + suffix;
      } else {
        el.textContent = current.toFixed(1) + suffix;
      }

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    }
    requestAnimationFrame(update);
  });
}

/* ============================================
   HELPER: Lightbox for gallery images
   ============================================ */
function setupLightbox(gallery) {
  // Create lightbox overlay
  const overlay = document.createElement("div");
  overlay.className = "lightbox-overlay";
  overlay.id = "gallery-lightbox";
  overlay.innerHTML = `
    <button class="lightbox-close" id="lightbox-close-btn">✕</button>
    <img src="" alt="Enlarged view" id="lightbox-img" />
  `;
  document.body.appendChild(overlay);

  const lightboxImg = overlay.querySelector("#lightbox-img");

  // Open on gallery item click
  gallery.querySelectorAll(".gallery-item").forEach((item) => {
    item.addEventListener("click", () => {
      lightboxImg.src = item.dataset.src;
      overlay.classList.add("active");
    });
  });

  // Close lightbox
  overlay.querySelector("#lightbox-close-btn").addEventListener("click", () => {
    overlay.classList.remove("active");
  });

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) {
      overlay.classList.remove("active");
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      overlay.classList.remove("active");
    }
  });
}
