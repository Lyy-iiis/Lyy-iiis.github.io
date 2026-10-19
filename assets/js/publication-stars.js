(function () {
  "use strict";

  var cachePrefix = "publication-stars:v1:";
  var cacheLifetime = 60 * 60 * 1000;
  var requests = new Map();

  function readCache(repo) {
    try {
      var cached = JSON.parse(localStorage.getItem(cachePrefix + repo));
      if (cached && Number.isInteger(cached.count) && cached.count >= 0 &&
          Number.isFinite(cached.updatedAt)) {
        return cached;
      }
    } catch (error) {
      // Star counts also work when browser storage is unavailable.
    }
    return null;
  }

  function showCount(link, count) {
    var stars = link.querySelector("[data-star-count]");
    var formatted = count.toLocaleString("en-US");
    stars.querySelector(".publication-star-value").textContent = formatted;
    stars.hidden = false;
    link.setAttribute("aria-label", link.querySelector("span").textContent +
      " on GitHub, " + formatted + (count === 1 ? " star" : " stars"));
    link.title = link.dataset.githubRepo + " · " + formatted + " GitHub stars";
  }

  function fetchCount(repo) {
    if (!requests.has(repo)) {
      var controller = new AbortController();
      var timeout = setTimeout(function () { controller.abort(); }, 6000);
      var request = fetch("https://api.github.com/repos/" + repo, {
        headers: { Accept: "application/vnd.github+json" },
        signal: controller.signal,
        credentials: "omit"
      }).then(function (response) {
        if (!response.ok) throw new Error("Repository unavailable");
        return response.json();
      }).then(function (data) {
        if (!Number.isInteger(data.stargazers_count) || data.stargazers_count < 0) {
          throw new Error("Invalid star count");
        }
        try {
          localStorage.setItem(cachePrefix + repo, JSON.stringify({
            count: data.stargazers_count,
            updatedAt: Date.now()
          }));
        } catch (error) {
          // A failed cache write must not hide a successfully fetched count.
        }
        return data.stargazers_count;
      }).finally(function () {
        clearTimeout(timeout);
      });
      requests.set(repo, request);
    }
    return requests.get(repo);
  }

  document.querySelectorAll(".publication-link[data-github-repo]").forEach(function (link) {
    var repo = link.dataset.githubRepo;
    if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo)) return;

    var cached = readCache(repo);
    if (cached) showCount(link, cached.count);
    if (cached && Date.now() - cached.updatedAt < cacheLifetime) return;

    fetchCount(repo).then(function (count) {
      showCount(link, count);
    }).catch(function () {
      // Keep the normal Code link (or the last known count) on network errors.
    });
  });
}());
