(function () {
  "use strict";

  var sidebar = document.querySelector(".academic-home .sidebar");
  if (!sidebar) return;

  var desktop = window.matchMedia("(min-width: 1024px)");
  var frame = 0;

  function update() {
    frame = 0;
    var inset = parseFloat(window.getComputedStyle(sidebar).top) || 0;
    var fits = desktop.matches && sidebar.getBoundingClientRect().height +
      inset * 2 <= document.documentElement.clientHeight;
    sidebar.classList.toggle("sidebar--pinned", fits);
  }

  function scheduleUpdate() {
    if (!frame) frame = window.requestAnimationFrame(update);
  }

  window.addEventListener("resize", scheduleUpdate);
  sidebar.querySelectorAll("img").forEach(function (image) {
    image.addEventListener("load", scheduleUpdate);
  });

  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(scheduleUpdate).observe(sidebar);
  }
  if (document.fonts) document.fonts.ready.then(scheduleUpdate);

  update();
}());
