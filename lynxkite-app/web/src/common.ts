import axios from "axios";
import { type User, UserManager } from "oidc-client-ts";
import { useContext, useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router";
import useSWR, { type Fetcher } from "swr";
import { LynxKiteState } from "./workspace/LynxKiteState";
import { buildCategoryHierarchy, type Catalogs } from "./workspace/NodeSearch";

export type GlobalConfig = {
  assistant_available: boolean;
  authentication_issuer: string | null;
  authentication_audience: string | null;
  enterprise_available: boolean;
};

let cachedConfig: GlobalConfig | undefined;
let userManager: UserManager | null = null;
let userManagerKey: string | undefined;
let loginStarted = false;
let axiosInterceptorsInstalled = false;

export async function loadConfig() {
  if (cachedConfig) {
    throw new Error("loadConfig() has already been called. Call getConfig() instead.");
  }
  cachedConfig = await apiJson<GlobalConfig>("/api/config");
}
export function getConfig(): GlobalConfig {
  if (!cachedConfig) {
    throw new Error("Global configuration is not initialized.");
  }
  return cachedConfig;
}

async function getAccessToken(): Promise<string | null> {
  const manager = getUserManager();
  if (!manager) {
    return null;
  }
  const user = await manager.getUser();
  if (!user || user.expired) {
    return null;
  }
  return user.access_token;
}

function getUserManager() {
  const issuer = cachedConfig?.authentication_issuer;
  const audience = cachedConfig?.authentication_audience;
  if (!issuer || !audience) {
    return null;
  }
  const key = `${issuer}|${audience}`;
  if (userManager && userManagerKey === key) {
    return userManager;
  }
  userManager = new UserManager({
    authority: issuer,
    client_id: audience,
    redirect_uri: `${window.location.origin}/auth/callback`,
    response_type: "code",
    scope: "openid profile email",
  });
  userManagerKey = key;
  return userManager;
}

export async function triggerLogin() {
  const manager = getUserManager();
  if (!manager || loginStarted) {
    return;
  }
  loginStarted = true;
  try {
    await manager.signinRedirect({
      state: {
        returnTo: `${window.location.pathname}${window.location.search}${window.location.hash}`,
      },
    });
  } catch (_error) {
    loginStarted = false;
  }
}

export async function triggerLogout() {
  const manager = getUserManager();
  if (!manager) {
    return;
  }
  // Remove local user state before redirecting.
  await manager.removeUser();
  // Build the logout URL manually with client_id instead of id_token_hint.
  // Keycloak 26 skips the confirmation page when id_token_hint is present,
  // but shows it when relying on the browser SSO session cookie + client_id.
  const metadata = await manager.metadataService.getMetadata();
  const endSessionEndpoint = metadata.end_session_endpoint;
  if (!endSessionEndpoint) return;
  const params = new URLSearchParams({
    client_id: cachedConfig?.authentication_audience ?? "",
    post_logout_redirect_uri: window.location.origin,
  });
  window.location.href = `${endSessionEndpoint}?${params}`;
}

function ensureAxiosInterceptors() {
  if (axiosInterceptorsInstalled) {
    return;
  }
  axios.interceptors.request.use(async (config) => {
    const token = await getAccessToken();
    if (token) {
      config.headers = config.headers || {};
      (config.headers as Record<string, string>).Authorization = `Bearer ${token}`;
    }
    return config;
  });
  axios.interceptors.response.use(
    (response) => response,
    async (error) => {
      if (error.response?.status === 401) {
        await triggerLogin();
      }
      return Promise.reject(error);
    },
  );
  axiosInterceptorsInstalled = true;
}

ensureAxiosInterceptors();

export async function apiFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  const headers = new Headers(init?.headers);
  const token = await getAccessToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(input, {
    ...init,
    headers,
  });
  if (response.status === 401) {
    await triggerLogin();
    throw new Error("Unauthorized");
  }
  return response;
}

export async function apiJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const response = await apiFetch(input, init);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function completeLoginCallback(): Promise<string> {
  const manager = getUserManager();
  if (!manager) {
    return "/";
  }
  const user = await manager.signinCallback();
  loginStarted = false;
  const state = user?.state as { returnTo?: string } | undefined;
  return state?.returnTo || "/";
}

export function useAuth() {
  const config = getConfig();
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => {
    const manager = getUserManager();
    if (!manager) {
      return;
    }
    manager.getUser().then(setUser);
    const onUserLoaded = (u: User) => setUser(u);
    const onUserUnloaded = () => setUser(null);
    manager.events.addUserLoaded(onUserLoaded);
    manager.events.addUserUnloaded(onUserUnloaded);
    return () => {
      manager.events.removeUserLoaded(onUserLoaded);
      manager.events.removeUserUnloaded(onUserUnloaded);
    };
  }, [config.authentication_issuer, config.authentication_audience]);
  return user;
}

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
    apiJson<Catalogs>(resource, init);
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

export const pathFetcher = <T>(url: string): Promise<T> => apiJson<T>(url);

export function parentPath(path: string): string {
  const parts = path.split("/").filter(Boolean);
  return parts.slice(0, -1).join("/");
}

export function shortName(path: string): string {
  return path.split("/").pop() ?? path;
}

export async function uploadFile(
  file: File,
  opts?: {
    onProgress?: (percent: number) => void;
  },
): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);
  await axios.post("/api/upload", formData, {
    onUploadProgress: (event) => {
      if (!opts?.onProgress) return;
      if (!event.total) return;
      const percent = Math.round((100 * event.loaded) / event.total);
      opts.onProgress(percent);
    },
  });
}

export const COLORS: { [key: string]: string } = {
  gray: "oklch(75% 0 0)",
  pink: "oklch(75% 0.2 0)",
  orange: "oklch(75% 0.2 55)",
  yellow: "oklch(90% 0.2 100)",
  green: "oklch(75% 0.2 130)",
  blue: "oklch(75% 0.2 230)",
  purple: "oklch(75% 0.2 290)",
  red: "oklch(75% 0.25 30)",
};

export const COLORS_MUTED: { [key: string]: string } = {
  gray: "oklch(90% 0 0)",
  pink: "oklch(90% 0.1 0)",
  orange: "oklch(90% 0.1 55)",
  yellow: "oklch(90% 0.1 100)",
  green: "oklch(90% 0.1 130)",
  blue: "oklch(90% 0.1 230)",
  purple: "oklch(90% 0.1 290)",
  red: "oklch(90% 0.15 30)",
};
