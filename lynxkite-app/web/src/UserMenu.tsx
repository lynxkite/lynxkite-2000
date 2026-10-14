// Dropdown user menu showing login/logout controls and the current user's name and email.
import { memo, useEffect, useRef, useState } from "react";
import LoginIcon from "~icons/tabler/login";
import LogoutIcon from "~icons/tabler/logout";
import UserCircleIcon from "~icons/tabler/user-circle";
import { getConfig, triggerLogin, triggerLogout, useAuth } from "./common";

// Re-rendering icons is expensive in dev mode; memoizing prevents it.
const Login = memo(LoginIcon);
const Logout = memo(LogoutIcon);
const UserCircle = memo(UserCircleIcon);

export default function UserMenu() {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const config = getConfig();
  const user = useAuth();
  const authEnabled = !!config?.authentication_issuer;
  const loggedIn = !!user && !user.expired;
  const userName = user?.profile?.name || "User";
  const userEmail =
    user?.profile?.email || user?.profile?.preferred_username || user?.profile?.name;

  useEffect(() => {
    if (!open) {
      return;
    }
    const onPointerDown = (event: PointerEvent) => {
      if (rootRef.current?.contains(event.target as Node)) {
        return;
      }
      setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  if (!authEnabled) {
    return null;
  }

  if (!loggedIn) {
    return (
      <div className="user-menu">
        <button
          type="button"
          className="user-menu-button"
          onClick={() => void triggerLogin()}
          title="Sign In"
        >
          <Login /> <span className="user-menu-label">Sign In</span>
        </button>
      </div>
    );
  }

  return (
    <div ref={rootRef} className={`user-menu dropdown dropdown-end ${open ? "dropdown-open" : ""}`}>
      <button
        type="button"
        className="user-menu-button"
        onClick={() => setOpen((value) => !value)}
        title={userName}
      >
        <UserCircle />
        <span className="user-menu-label">{userName}</span>
      </button>
      {open && (
        <ul className="dropdown-content menu shadow-lg rounded-box bg-base-100 z-50 w-52 p-2 mt-2 end-0">
          <li className="menu-title px-4 py-2 text-sm opacity-70">{userEmail}</li>
          <li>
            <button type="button" onClick={() => void triggerLogout()}>
              <Logout /> Log out
            </button>
          </li>
        </ul>
      )}
    </div>
  );
}
