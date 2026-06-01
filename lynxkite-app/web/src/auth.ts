// Keycloak OIDC authentication utilities using Authorization Code flow with PKCE.

const KEYCLOAK_URL = "http://localhost:8080";
const REALM = "lynxkite-dev";
const CLIENT_ID = "lynxkite-web";
const REDIRECT_URI = `${window.location.origin}/auth/callback`;
const TOKEN_STORAGE_KEY = "lynxkite-auth-token";
const USER_STORAGE_KEY = "lynxkite-auth-user";
const VERIFIER_STORAGE_KEY = "lynxkite-pkce-verifier";

const BASE_URL = `${KEYCLOAK_URL}/realms/${REALM}/protocol/openid-connect`;

export interface AuthUser {
  name: string;
  email?: string;
  preferred_username?: string;
}

function generateRandomString(length: number): string {
  const array = new Uint8Array(length);
  crypto.getRandomValues(array);
  return Array.from(array, (b) => b.toString(36).padStart(2, "0"))
    .join("")
    .slice(0, length);
}

async function sha256(plain: string): Promise<ArrayBuffer> {
  const encoder = new TextEncoder();
  return crypto.subtle.digest("SHA-256", encoder.encode(plain));
}

function base64UrlEncode(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

async function buildAuthUrl(action: "auth" | "registrations"): Promise<string> {
  const verifier = generateRandomString(64);
  sessionStorage.setItem(VERIFIER_STORAGE_KEY, verifier);

  const challengeBuffer = await sha256(verifier);
  const challenge = base64UrlEncode(challengeBuffer);

  const params = new URLSearchParams({
    client_id: CLIENT_ID,
    response_type: "code",
    redirect_uri: REDIRECT_URI,
    scope: "openid profile email",
    code_challenge: challenge,
    code_challenge_method: "S256",
  });

  return `${BASE_URL}/${action}?${params}`;
}

export async function login(): Promise<void> {
  window.location.href = await buildAuthUrl("auth");
}

export async function register(): Promise<void> {
  window.location.href = await buildAuthUrl("registrations");
}

export async function handleCallback(): Promise<boolean> {
  const params = new URLSearchParams(window.location.search);
  const code = params.get("code");
  if (!code) return false;

  const verifier = sessionStorage.getItem(VERIFIER_STORAGE_KEY);
  if (!verifier) return false;

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: CLIENT_ID,
    code,
    redirect_uri: REDIRECT_URI,
    code_verifier: verifier,
  });

  const res = await fetch(`${BASE_URL}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body,
  });

  if (!res.ok) return false;

  const data = await res.json();
  sessionStorage.removeItem(VERIFIER_STORAGE_KEY);
  localStorage.setItem(TOKEN_STORAGE_KEY, data.access_token);

  // Decode the JWT payload to get user info.
  const payload = JSON.parse(atob(data.access_token.split(".")[1]));
  const user: AuthUser = {
    name: payload.name || payload.preferred_username || "User",
    email: payload.email,
    preferred_username: payload.preferred_username,
  };
  localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(user));

  return true;
}

export function logout(): void {
  const params = new URLSearchParams({
    client_id: CLIENT_ID,
    post_logout_redirect_uri: `${window.location.origin}/auth/logged-out`,
  });

  window.location.href = `${BASE_URL}/logout?${params}`;
}

export function clearAuthState(): void {
  localStorage.removeItem(TOKEN_STORAGE_KEY);
  localStorage.removeItem(USER_STORAGE_KEY);
}

export function getUser(): AuthUser | null {
  const stored = localStorage.getItem(USER_STORAGE_KEY);
  if (!stored) return null;
  try {
    return JSON.parse(stored);
  } catch {
    return null;
  }
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_STORAGE_KEY);
}

export function isLoggedIn(): boolean {
  return !!getToken();
}
