import DOMPurify from "dompurify";
import { memo } from "react";

interface InlineSvgProps {
  svg: string;
  className?: string;
  [key: string]: any;
}

const InlineSvg = memo(function InlineSvg({ svg, className, ...props }: InlineSvgProps) {
  return (
    <span
      className={className}
      {...props}
      dangerouslySetInnerHTML={{ __html: DOMPurify.sanitize(svg) }}
    />
  );
});

export default InlineSvg;
