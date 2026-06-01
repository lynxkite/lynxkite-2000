import { useState } from "react";
import Login from "~icons/tabler/login";
import Logout from "~icons/tabler/logout";
import UserCircle from "~icons/tabler/user-circle";
import { getUser, isLoggedIn, login, logout } from "./auth";

export default function UserMenu() {
  const [open, setOpen] = useState(false);
  const loggedIn = isLoggedIn();
  const user = getUser();

  if (!loggedIn) {
    return (
      <button type="button" className="user-menu-button" onClick={() => login()} title="Log in">
        <Login /> <span className="user-menu-label">Log in</span>
      </button>
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
