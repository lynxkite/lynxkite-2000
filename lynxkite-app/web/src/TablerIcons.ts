import rawTablerIcons from "@iconify-json/tabler/icons.json";

type IconData = {
  body: string;
  width?: number;
  height?: number;
};

type IconCollection = {
  width?: number;
  height?: number;
  icons: Record<string, IconData>;
};

const tablerIcons = rawTablerIcons as IconCollection;

export function getTablerIconSvgMarkup(name: string): string {
  const icon = tablerIcons.icons[name];
  if (!icon) {
    console.warn(`Missing Tabler icon: ${name}`);
    return getTablerIconSvgMarkup("exclamation-circle");
  }
  const width = icon.width ?? tablerIcons.width ?? 24;
  const height = icon.height ?? tablerIcons.height ?? 24;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}">${icon.body}</svg>`;
}
