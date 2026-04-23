// This component handles the callback from the authentication provider (e.g., Keycloak).
import { useEffect, useState } from "react";
import { useNavigate } from "react-router";
import { completeLoginCallback, loadConfig } from "./common.ts";

export default function AuthCallback() {
  const navigate = useNavigate();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const run = async () => {
      try {
        await loadConfig();
        const returnTo = await completeLoginCallback();
        navigate(returnTo, { replace: true });
      } catch (e) {
        const message = e instanceof Error ? e.message : "Authentication callback failed.";
        setError(message);
      }
    };
    run();
  }, [navigate]);

  if (error) {
    return <div className="p-4">Authentication failed: {error}</div>;
  }

  return <div className="p-4">Signing you in...</div>;
}
