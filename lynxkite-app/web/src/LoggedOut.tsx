import { useEffect } from "react";
import { useNavigate } from "react-router";
import { clearAuthState } from "./auth";

export default function LoggedOut() {
  const navigate = useNavigate();

  useEffect(() => {
    clearAuthState();
    navigate("/", { replace: true });
  }, [navigate]);

  return (
    <div
      style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh" }}
    >
      <span className="loading loading-spinner loading-lg" />
    </div>
  );
}
