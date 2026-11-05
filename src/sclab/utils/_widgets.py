import traceback

import anywidget
from ipywidgets import Dropdown, Output
from traitlets import Unicode


def show_outputs(output: Output, *bundles: dict) -> None:
    """Replace what ``output`` shows by ``bundles`` (mime type -> data each).

    Set from the kernel, unlike ``with output: display(...)``: the frontend
    then sends every captured output back to the kernel, which echoes it to
    the frontends, so a large table crosses the websocket three times. The
    obs table of 16k cells is 7 MB: the copy sent by the browser delays its
    pongs, clicks and execute requests until the connection times out on a
    slow link (the dashboard looks hung).
    """
    output.outputs = tuple(
        {"output_type": "display_data", "data": data, "metadata": {}}
        for data in bundles
    )


class LazyOutput(Output):
    """Output built by ``render()`` (an iterable of bundles, see :func:`show_outputs`)
    only while it is shown.

    It is shown while every ``(container, child)`` given to :meth:`show_when` has
    ``child`` as the container's selected page (Tab, Stack, Accordion); with no
    pairs it is always shown. Until :meth:`show_when` is called it is hidden.
    :meth:`invalidate` (the data changed) renders again now if shown, else on the
    next show. With ``unload``, hiding clears it, so a page reload does not
    download it again as widget state; the rendered bundles are kept, so showing
    it again re-sends them without rendering.
    """

    def __init__(self, render, unload: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._render = render
        self.unload = unload
        self.shown = False
        self.stale = True
        self._bundles = ()
        self._path = ()

    def show_when(self, *path) -> None:
        for container, _ in self._path:
            container.unobserve(self._update_shown, ["selected_index", "children"])
        self._path = path
        for container, _ in path:
            container.observe(self._update_shown, ["selected_index", "children"])
        self._update_shown()

    def invalidate(self) -> None:
        self.stale = True
        if self.shown:
            self._show()

    def _update_shown(self, change=None) -> None:
        shown = all(
            c.selected_index is not None
            and c.selected_index < len(c.children)
            and c.children[c.selected_index] is child
            for c, child in self._path
        )
        if shown == self.shown:
            return
        self.shown = shown
        if shown:
            self._show()
        elif self.unload:
            self.outputs = ()

    def _show(self) -> None:
        if self.stale:
            try:
                self._bundles = tuple(self._render())
                self.stale = False
            except Exception:  # shown in place of the content; retried on next show
                self._bundles = ({"text/plain": traceback.format_exc()},)
            show_outputs(self, *self._bundles)
        elif not self.outputs:
            show_outputs(self, *self._bundles)


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
        model.on("msg:custom", ({ text, filename, type }, buffers) => {
          const a = document.createElement("a");
          const parts = buffers && buffers.length ? buffers : [text];
          a.href = URL.createObjectURL(new Blob(parts, { type }));
          a.download = filename;
          a.click();
          setTimeout(() => URL.revokeObjectURL(a.href), 1000);
        });
      },
    };
    """

    def save(self, text: str | bytes, filename: str, type: str = "text/html"):
        if isinstance(text, bytes):  # e.g. a PNG: a binary buffer, not JSON
            self.send({"filename": filename, "type": type}, buffers=[text])
        else:
            self.send({"text": text, "filename": filename, "type": type})
