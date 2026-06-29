/**
 * Interactive project timeline component.
 */

const MILESTONES = [
  {
    date: "June 2026 — Week 1",
    icon: "📋",
    title: "Project Planning",
    description:
      "Problem definition, literature review, dataset sourcing, and selection of YOLOv8s-P2 architecture for real-time UAV-based early wildfire detection.",
  },
  {
    date: "June 2026 — Week 2",
    icon: "📦",
    title: "Data Preparation",
    description:
      "Processed 21,527 source images through the Early-Stage Fire Simulation Augmentation (EFSA) pipeline, synthesizing 13,862 high-quality early-stage training images.",
  },
  {
    date: "June 21, 2026",
    icon: "🚀",
    title: "Phase 1 — Model Training",
    description:
      "Trained YOLOv8s augmented with a P2 detection head for 100 epochs on Kaggle (2× Tesla T4 GPUs). Achieved 97.7% mAP@50 and 90.0% mAP@50-95 on the EFSA test set.",
  },
  {
    date: "June 26, 2026",
    icon: "🔄",
    title: "Phase 2 — Fine-tuning",
    description:
      "Resumed training with optimized hyperparameters for further convergence. Evaluated real-world generalization on unseen incipient fire footage.",
  },
  {
    date: "June 27, 2026",
    icon: "🌐",
    title: "Web Demo & Deployment",
    description:
      "Exported model to ONNX format for optimized inference. Built Flask API backend and Vite + Tailwind CSS frontend. Deployed to Google Cloud Run & Firebase Hosting.",
  },
];

export function createTimeline() {
  const section = document.createElement("section");
  section.className = "py-20 px-6";
  section.id = "timeline-section";

  section.innerHTML = `
    <div class="max-w-5xl mx-auto">
      <h2 class="section-title text-center gradient-text">Project Timeline</h2>
      <p class="section-subtitle text-center">From concept to deployment</p>

      <div class="relative mt-12">
        <!-- Center line -->
        <div class="timeline-line hidden md:block"></div>

        <!-- Mobile line (left) -->
        <div class="md:hidden absolute left-5 top-0 bottom-0 w-0.5 bg-gradient-to-b from-accent-dark via-accent-light to-accent-dark"></div>

        <div class="space-y-12 md:space-y-16" id="timeline-items">
          ${MILESTONES.map((m, i) => createTimelineItem(m, i)).join("")}
        </div>
      </div>
    </div>
  `;

  // Animate items on scroll
  requestAnimationFrame(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add(
              entry.target.dataset.dir === "left"
                ? "animate-slide-in-left"
                : "animate-slide-in-right"
            );
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.15 }
    );

    section.querySelectorAll(".timeline-item").forEach((el) => {
      observer.observe(el);
    });
  });

  return section;
}

function createTimelineItem(milestone, index) {
  const isLeft = index % 2 === 0;

  return `
    <div class="timeline-item opacity-0 relative flex items-start gap-4 md:gap-0"
         data-dir="${isLeft ? "left" : "right"}">

      <!-- Mobile layout -->
      <div class="md:hidden flex-shrink-0 mt-1">
        <div class="timeline-dot"></div>
      </div>
      <div class="md:hidden glass-card p-5 flex-1">
        <div class="flex items-center gap-2 mb-2">
          <span class="text-2xl">${milestone.icon}</span>
          <span class="text-xs font-mono text-accent-light">${milestone.date}</span>
        </div>
        <h3 class="font-display font-semibold text-lg text-text-primary mb-2">${milestone.title}</h3>
        <p class="text-sm text-text-secondary leading-relaxed">${milestone.description}</p>
      </div>

      <!-- Desktop layout -->
      <div class="hidden md:grid md:grid-cols-[1fr_auto_1fr] md:gap-8 w-full items-center">
        <!-- Left content -->
        <div class="${isLeft ? "" : "order-3"}">
          ${
            isLeft
              ? `<div class="glass-card p-6 text-right">
              <div class="flex items-center justify-end gap-2 mb-2">
                <span class="text-xs font-mono text-accent-light">${milestone.date}</span>
                <span class="text-2xl">${milestone.icon}</span>
              </div>
              <h3 class="font-display font-semibold text-lg text-text-primary mb-2">${milestone.title}</h3>
              <p class="text-sm text-text-secondary leading-relaxed">${milestone.description}</p>
            </div>`
              : ""
          }
        </div>

        <!-- Center dot -->
        <div class="flex justify-center">
          <div class="timeline-dot"></div>
        </div>

        <!-- Right content -->
        <div class="${isLeft ? "order-3" : ""}">
          ${
            !isLeft
              ? `<div class="glass-card p-6">
              <div class="flex items-center gap-2 mb-2">
                <span class="text-2xl">${milestone.icon}</span>
                <span class="text-xs font-mono text-accent-light">${milestone.date}</span>
              </div>
              <h3 class="font-display font-semibold text-lg text-text-primary mb-2">${milestone.title}</h3>
              <p class="text-sm text-text-secondary leading-relaxed">${milestone.description}</p>
            </div>`
              : ""
          }
        </div>
      </div>
    </div>
  `;
}
