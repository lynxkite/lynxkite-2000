import { useLocation } from "react-router";

export function usePath() {
  // Decode special characters. Drop trailing slash. (Some clients add it, e.g. Playwright.)
  const path = decodeURIComponent(useLocation().pathname).replace(/[/]$/, "");
  return path;
}

export const COLORS: { [key: string]: string } = {
  gray: "oklch(95% 0 0)",
  pink: "oklch(70% 0.15 0)",
  orange: "oklch(70% 0.15 55)",
  green: "oklch(70% 0.15 150)",
  blue: "oklch(70% 0.15 230)",
  purple: "oklch(70% 0.15 290)",
};
