import DOMPurify from "dompurify";
import { useEffect, useState, memo } from "react";

interface InlineSvgProps {
  src?: string;
  className?: string;
  [key: string]: any;
}

const InlineSvg = memo(function InlineSvg({ src, className, ...props }: InlineSvgProps) {
  const [svg, setSvg] = useState<string | null>(null);
  useEffect(() => {
    if (!src) return;
    fetch(src)
      .then((res) => res.text())
      .then((text) => setSvg(text))
      .catch((err) => console.error("Error loading SVG:", err));
  }, [src]);
  return (
    <span
      className={className}
      {...props}
      dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(svg || "") }}
    />
  );
});

export default InlineSvg;
