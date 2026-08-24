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
const warnedMissingIcons = new Set<string>();

function normalizeIconName(name: string): string {
  return name.startsWith("tabler:") ? name.slice("tabler:".length) : name;
}

export function getTablerIconSvgMarkup(name: string): string | undefined {
  const normalizedName = normalizeIconName(name);
  const icon = tablerIcons.icons[normalizedName];
  if (!icon) {
    if (import.meta.env.DEV && !warnedMissingIcons.has(normalizedName)) {
      warnedMissingIcons.add(normalizedName);
      console.warn(`Missing Tabler icon: ${normalizedName}`);
    }
    return undefined;
  }
  const width = icon.width ?? tablerIcons.width ?? 24;
  const height = icon.height ?? tablerIcons.height ?? 24;
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}">${icon.body}</svg>`;
}
