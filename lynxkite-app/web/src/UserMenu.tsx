import { useState } from "react";
import Login from "~icons/tabler/login";
import Logout from "~icons/tabler/logout";
import UserCircle from "~icons/tabler/user-circle";
import UserPlus from "~icons/tabler/user-plus";
import { getUser, isLoggedIn, login, logout, register } from "./auth";

export default function UserMenu() {
  const [open, setOpen] = useState(false);
  const loggedIn = isLoggedIn();
  const user = getUser();

  if (!loggedIn) {
    return (
      <div className="user-menu-actions">
        <button type="button" className="user-menu-button" onClick={() => login()} title="Sign In">
          <Login /> <span className="user-menu-label">Sign In</span>
        </button>
        <button
          type="button"
          className="user-menu-button"
          onClick={() => register()}
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
        title={user?.name || "User"}
      >
        <UserCircle />
        <span className="user-menu-label">{user?.name || "User"}</span>
      </button>
      {open && (
        <ul className="dropdown-content menu shadow-lg rounded-box bg-base-100 z-50 w-52 p-2 mt-2">
          <li className="menu-title px-4 py-2 text-sm opacity-70">
            {user?.email || user?.preferred_username || user?.name}
          </li>
          <li>
            <button type="button" onClick={() => logout()}>
              <Logout /> Log out
            </button>
          </li>
        </ul>
      )}
    </div>
  );
}
