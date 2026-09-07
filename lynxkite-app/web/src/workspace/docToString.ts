export function docToString(doc: any): string {
  if (!doc) return "";
  return (
    doc.map?.((section: any) => (section.kind === "text" ? section.value : "")).join("\n") ??
    String(doc)
  );
}
