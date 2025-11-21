import { useEffect, useState } from "react";

interface InlineSvgProps {
  src?: string;
  className?: string;
  [key: string]: any;
}

export default function InlineSvg({ src, className, ...props }: InlineSvgProps) {
  const [svg, setSvg] = useState<string | null>(null);
  useEffect(() => {
    if (!src) return;
    fetch(src)
      .then((res) => res.text())
      .then((text) => setSvg(text))
      .catch((err) => console.error("Error loading SVG:", err));
  }, [src]);
  return <span className={className} {...props} dangerouslySetInnerHTML={{ __html: svg || "" }} />;
}
