import { useState } from "react";
import Login from "~icons/tabler/login";
import Logout from "~icons/tabler/logout";
import UserCircle from "~icons/tabler/user-circle";
import UserPlus from "~icons/tabler/user-plus";
import { triggerLogin, triggerLogout, triggerRegister, useAuth } from "./common";

export default function UserMenu() {
  const [open, setOpen] = useState(false);
  const user = useAuth();
  const loggedIn = !!user && !user.expired;
  const userName = user?.profile?.name || "User";
  const userEmail =
    user?.profile?.email || user?.profile?.preferred_username || user?.profile?.name;

  if (!loggedIn) {
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
        <button
          type="button"
          className="user-menu-button"
          onClick={() => void triggerRegister()}
          title="Sign Up"
        >
          <UserPlus /> <span className="user-menu-label">Sign Up</span>
        </button>
      </div>
    );
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
