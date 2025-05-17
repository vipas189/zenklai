// Hide header on scroll down
let lastScrollPosition = 0;
const header = document.getElementById("header");
const scrollableElement = document.documentElement;

window.addEventListener("scroll", () => {
  const currentScrollPosition = scrollableElement.scrollTop;
  if (
    currentScrollPosition > lastScrollPosition &&
    currentScrollPosition > header.offsetHeight
  ) {
    header.classList.add("hidden");
  } else {
    header.classList.remove("hidden");
  }
  lastScrollPosition = currentScrollPosition <= 0 ? 0 : currentScrollPosition;
});

// Tab functionality
function openTab(tabName) {
  const tabsContainer = document.getElementById("tabs-container");
  const trainTabContent = document.getElementById("train");
  const testTabContent = document.getElementById("test");
  const trainBtn = document.getElementById("train-btn");
  const testBtn = document.getElementById("test-btn");

  trainTabContent.classList.remove("active");
  testTabContent.classList.remove("active");
  trainBtn.classList.remove("active");
  testBtn.classList.remove("active");

  tabsContainer.classList.remove("train-mode", "test-mode");

  if (tabName === "train") {
    trainTabContent.classList.add("active");
    trainBtn.classList.add("active");
    tabsContainer.classList.add("train-mode");
  } else if (tabName === "test") {
    testTabContent.classList.add("active");
    testBtn.classList.add("active");
    tabsContainer.classList.add("test-mode");
  }
}

// Function to update file label (simple version)
function updateFileLabel(inputId, labelId) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  if (input.files && input.files.length > 0) {
    if (input.webkitdirectory) {
      // For folder selection
      label.textContent = input.files.length + " items in folder";
    } else {
      // For single file
      label.textContent = input.files[0].name;
    }
  } else {
    label.textContent = input.webkitdirectory
      ? "No folder chosen"
      : "No file chosen";
  }
}

// Initialize with the first tab open
window.onload = function () {
  openTab("train");
  // Set initial file labels correctly if needed (though onchange handles it after interaction)
  updateFileLabel("train-file-input", "train-file-label");
  updateFileLabel("test-file-input", "test-file-label");
};
