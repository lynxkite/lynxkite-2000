import { useLocation } from "react-router";

export function usePath() {
  // Decode special characters. Drop trailing slash. (Some clients add it, e.g. Playwright.)
  const path = decodeURIComponent(useLocation().pathname).replace(/[/]$/, "");
  return path;
}

export const COLORS: { [key: string]: string } = {
  gray: "oklch(95% 0 0)",
  pink: "oklch(75% 0.2 0)",
  orange: "oklch(75% 0.2 55)",
  green: "oklch(75% 0.2 150)",
  blue: "oklch(75% 0.2 230)",
  purple: "oklch(75% 0.2 290)",
};
