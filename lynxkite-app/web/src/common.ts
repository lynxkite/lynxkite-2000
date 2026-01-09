import { useContext, useMemo } from "react";
import { useLocation } from "react-router";
import useSWR, { type Fetcher } from "swr";
import { LynxKiteState } from "./workspace/LynxKiteState";
import { buildCategoryHierarchy, type Catalogs } from "./workspace/NodeSearch";

export function usePath() {
  // Decode special characters. Drop trailing slash. (Some clients add it, e.g. Playwright.)
  const path = decodeURIComponent(useLocation().pathname).replace(/[/]$/, "");
  return path;
}

export function useCategoryHierarchy() {
  const ws = useContext(LynxKiteState).workspace;
  const env = ws?.env;
  const path = usePath().replace(/^[/]edit[/]/, "");
  const fetcher: Fetcher<Catalogs> = (resource: string, init?: RequestInit) =>
    fetch(resource, init).then((res) => res.json());
  const encodedPathForAPI = path!
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  const catalog = useSWR(`/api/catalog?workspace=${encodedPathForAPI}`, fetcher);
  const categoryHierarchy = useMemo(() => {
    if (!catalog.data || !env) return undefined;
    return buildCategoryHierarchy(catalog.data[env]);
  }, [catalog, env]);
  return categoryHierarchy;
}

export const COLORS: { [key: string]: string } = {
  gray: "oklch(95% 0 0)",
  pink: "oklch(70% 0.15 0)",
  orange: "oklch(70% 0.15 55)",
  green: "oklch(70% 0.15 150)",
  blue: "oklch(70% 0.15 230)",
  purple: "oklch(70% 0.15 290)",
  red: "oklch(70% 0.25 30)",
};
