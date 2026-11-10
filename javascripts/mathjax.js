// MathJax configuration for the formulas written with pymdownx.arithmatex (generic mode, see mkdocs.yml).
// Arithmatex wraps every formula ($...$ inline, $$...$$ as a block) in an element with the class `arithmatex`;
// MathJax only typesets those elements. Loaded before MathJax itself (see extra_javascript in mkdocs.yml).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

// Material's instant navigation loads pages without a full reload: typeset the formulas again on every page.
document$.subscribe(() => {
  MathJax.startup.output.clearCache();
  MathJax.typesetClear();
  MathJax.texReset();
  MathJax.typesetPromise();
});
