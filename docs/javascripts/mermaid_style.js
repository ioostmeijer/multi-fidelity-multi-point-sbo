// Mermaid diagrams in the style of the draw.io diagrams in docs/assets.
// Material draws Mermaid in a closed shadow DOM and passes its own CSS as `themeCSS` to mermaid.initialize. This script
// (loaded after Mermaid, see extra_javascript in mkdocs.yml) appends the CSS below, which colours the nodes by their
// Mermaid class (`class X config` in a diagram). The colours are CSS variables from
// docs/stylesheets/extra.css, so they follow the light/dark toggle.
(function () {
  const kinds = ["user", "config", "sbo", "optimisation", "surrogate", "infill"];
  const css = kinds
    .map(
      (k) =>
        `.node.${k} rect, .node.${k} path, .node.${k} polygon {` +
        ` fill: var(--dg-${k}-fill) !important; stroke: var(--dg-${k}-stroke) !important; stroke-width: 1px !important; }` +
        ` .node.${k} .nodeLabel { color: var(--dg-text) !important; }`
    )
    .join("\n");

  const mermaidInitialize = mermaid.initialize.bind(mermaid);
  mermaid.initialize = (config) => mermaidInitialize({ ...config, themeCSS: (config.themeCSS || "") + css });
})();
