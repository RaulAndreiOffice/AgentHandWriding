// Reads a JSON array of {tex, display} from stdin and writes a JSON array of the
// same length: null when KaTeX parses the expression, else the error message.
// Used by app/services/verifier.py; one process checks a whole page.
import katex from "katex";

let input = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (input += chunk));
process.stdin.on("end", () => {
  const items = JSON.parse(input || "[]");
  const out = items.map(({ tex, display }) => {
    try {
      katex.renderToString(tex, { displayMode: !!display, throwOnError: true, strict: "ignore" });
      return null;
    } catch (err) {
      return String(err.message || err).replace(/^KaTeX parse error: /, "");
    }
  });
  process.stdout.write(JSON.stringify(out));
});
