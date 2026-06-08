// Dropdown user menu showing login/logout controls and the current user's name and email.
import { memo, useState } from "react";
import LoginIcon from "~icons/tabler/login";
import LogoutIcon from "~icons/tabler/logout";
import UserCircleIcon from "~icons/tabler/user-circle";
import { triggerLogin, triggerLogout, useAuth, useConfig } from "./common";

// Re-rendering icons is expensive in dev mode; memoizing prevents it.
const Login = memo(LoginIcon);
const Logout = memo(LogoutIcon);
const UserCircle = memo(UserCircleIcon);

export default function UserMenu() {
  const [open, setOpen] = useState(false);
  const { data: config } = useConfig();
  const user = useAuth();
  const authEnabled = !!config?.authentication_issuer;
  const loggedIn = !!user && !user.expired;
  const userName = user?.profile?.name || "User";
  const userEmail =
    user?.profile?.email || user?.profile?.preferred_username || user?.profile?.name;

  if (!loggedIn && authEnabled) {
    return (
      <div className="user-menu-actions">
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

  if (!loggedIn) {
    return null;
  }

  return (
    <div className={`dropdown dropdown-end ${open ? "dropdown-open" : ""}`}>
      <button
        type="button"
        className="user-menu-button"
        onClick={() => setOpen(!open)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        title={userName}
      >
        <UserCircle />
        <span className="user-menu-label">{userName}</span>
      </button>
      {open && (
        <ul className="dropdown-content menu shadow-lg rounded-box bg-base-100 z-50 w-52 p-2 mt-2">
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
