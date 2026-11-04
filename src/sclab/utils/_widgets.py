import anywidget
from ipywidgets import Dropdown
from traitlets import Unicode


def set_options_keep_value(dropdown: Dropdown, options) -> None:
    """Replace ``dropdown.options``, keeping the current value if still offered.

    ipywidgets resets the selection to the first option whenever the options
    change. The restore happens inside ``hold_sync`` so the frontend receives
    one atomic update (options and selected index) instead of the transient
    reset, which it can render as an empty dropdown.
    """
    old_value = dropdown.value
    available = options.values() if isinstance(options, dict) else options
    with dropdown.hold_sync():
        dropdown.options = options
        if old_value in available:
            dropdown.value = old_value


class ResizeNotifier(anywidget.AnyWidget):
    """Zero-height, full-width widget that fires a window ``resize`` event.

    It fires once its first size is laid out and again whenever its width
    changes. Put it next to a responsive plotly figure: the plotly>=6 widget
    renders while still detached (so it falls back to plotly.js's 700px
    default) and, unlike the plotly 5 view, never re-measures on attach/resize
    messages. plotly.js only re-measures on ``resize`` when ``responsive`` is set.
    """

    _esm = """
    export default {
      render({ el }) {
        el.style.cssText = "width:100%;height:0";
        let raf = 0;
        const ro = new ResizeObserver(() => {
          cancelAnimationFrame(raf);
          raf = requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
        });
        ro.observe(el);
        return () => ro.disconnect();
      },
    };
    """


class CodeActions(anywidget.AnyWidget):
    """Copy code and Download .py buttons for the script in ``text``.

    Copy uses the browser Clipboard API (needs a secure context: https or
    localhost); download saves ``text`` as ``filename`` via a Blob URL.
    """

    text = Unicode("").tag(sync=True)
    filename = Unicode("sclab_analysis.py").tag(sync=True)

    _esm = """
    export default {
      render({ model, el }) {
        const button = (label, onclick) => {
          const b = document.createElement("button");
          b.className = "jupyter-button widget-button";
          b.style.width = "auto";
          b.textContent = label;
          b.onclick = onclick;
          el.appendChild(b);
          return b;
        };
        const copy = button("Copy code", async () => {
          try {
            await navigator.clipboard.writeText(model.get("text"));
            copy.textContent = "Copied";
          } catch {
            copy.textContent = "Copy failed: select the code below";
          }
          setTimeout(() => (copy.textContent = "Copy code"), 2000);
        });
        button("Download .py", () => {
          const blob = new Blob([model.get("text")], { type: "text/x-python" });
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = model.get("filename");
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        });
      },
    };
    """


class Download(anywidget.AnyWidget):
    """Saves files in the browser: :meth:`save` triggers a Blob download.

    The content goes in a one-off message, not a synced trait, so it is not
    kept in the widget state (or the notebook). Nothing is written on the
    kernel's disk.
    """

    _esm = """
    export default {
      initialize({ model }) {
        model.on("msg:custom", ({ text, filename, type }) => {
          const a = document.createElement("a");
          a.href = URL.createObjectURL(new Blob([text], { type }));
          a.download = filename;
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        });
      },
    };
    """

    def save(self, text: str, filename: str, type: str = "text/html"):
        self.send({"text": text, "filename": filename, "type": type})
