const fadeElements = document.querySelectorAll(".fade-in");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if ("IntersectionObserver" in window && !reducedMotion) {
  document.documentElement.classList.add("reveal-ready");
  const fadeObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });

  fadeElements.forEach((element) => fadeObserver.observe(element));
} else {
  fadeElements.forEach((element) => element.classList.add("visible"));
}

const depthControl = document.getElementById("depthControl");
const rankControl = document.getElementById("rankControl");
const depthValue = document.getElementById("depthValue");
const rankValue = document.getElementById("rankValue");
const ambientValue = document.getElementById("ambientValue");
const reductionValue = document.getElementById("reductionValue");

function updateDimensionLab() {
  if (!depthControl || !rankControl || !depthValue || !rankValue ||
      !ambientValue || !reductionValue) return;

  const depth = Number(depthControl.value);
  const rank = Number(rankControl.value);
  const ambient = depth * (15 + 10);
  const reduction = 100 * (1 - rank / ambient);

  depthValue.value = String(depth);
  rankValue.value = String(rank);
  ambientValue.innerText = String(ambient);
  reductionValue.innerText = `${reduction.toFixed(1)}%`;
}

if (depthControl && rankControl) {
  depthControl.addEventListener("input", updateDimensionLab);
  rankControl.addEventListener("input", updateDimensionLab);
  updateDimensionLab();
}

const copyButton = document.getElementById("copyBibtex");
const bibtexCode = document.getElementById("bibtexCode");
const copyStatus = document.getElementById("copyStatus");

function legacyCopy(text) {
  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.appendChild(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("copy command was rejected");
}

async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  legacyCopy(text);
}

if (copyButton && bibtexCode && copyStatus) {
  copyButton.addEventListener("click", async () => {
    try {
      await copyText(bibtexCode.innerText);
      copyButton.innerText = "Copied";
      copyStatus.innerText = "BibTeX copied to the clipboard.";
    } catch (error) {
      copyButton.innerText = "Copy failed";
      copyStatus.innerText = "Clipboard access failed. Select and copy the BibTeX manually.";
    }

    window.setTimeout(() => {
      copyButton.innerText = "Copy BibTeX";
    }, 1800);
  });
}
