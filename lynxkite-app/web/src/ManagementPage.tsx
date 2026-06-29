// Pages like the directory browser use this to get a uniform layout.
import { useState } from "react";
import { NavLink } from "react-router";
import Activity from "~icons/tabler/activity";
import Book from "~icons/tabler/book";
import BrandGithub from "~icons/tabler/brand-github";
import Folder from "~icons/tabler/folder";
import LogOut from "~icons/tabler/logout";
import Menu from "~icons/tabler/menu-2";
import SettingsIcon from "~icons/tabler/settings";
import UserIcon from "~icons/tabler/user";
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";
import { triggerLogout, useConfig, useCurrentUser } from "./common.ts";

const NAV_ITEMS = [
  { to: "/dir", icon: <Folder />, label: "Files" },
  { to: "/progress", icon: <Activity />, label: "Progress" },
];

export default function ManagementPage(props: {
  children: React.ReactNode;
  className?: string;
  sidebar?: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const config = useConfig();
  const currentUser = useCurrentUser();
  const authEnabled = !!config.data?.authentication_issuer;

  return (
    <div className={`management-page ${props.className || ""}`}>
      <div className={`management-sidebar${sidebarOpen ? " open" : ""}`}>
        <button
          type="button"
          className="sidebar-toggle-btn"
          aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          <Menu />
        </button>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to + item.label}
              to={item.to}
              className={({ isActive }) => `sidebar-nav-item${isActive ? " active" : ""}`}
              title={item.label}
            >
              {item.icon}
              <span className="sidebar-label">{item.label}</span>
            </NavLink>
          ))}
          <a
            href="https://lynxkite.com/lynxkite-2000/"
            target="_blank"
            rel="noopener noreferrer"
            className="sidebar-nav-item"
            title="Documentation"
          >
            <Book />
            <span className="sidebar-label">Documentation</span>
          </a>
          <a className="sidebar-nav-item" title="Settings" href="#">
            <SettingsIcon />
            <span className="sidebar-label">Settings</span>
          </a>
        </nav>

        {props.sidebar && <div className="sidebar-inner sidebar-page-section">{props.sidebar}</div>}

        <div className="sidebar-footer">
          {authEnabled && (
            <div className="sidebar-user">
              <UserIcon className="sidebar-user-icon" />
              <span className="sidebar-label sidebar-user-name">{currentUser ?? "User"}</span>
              <button
                type="button"
                className="sidebar-logout-btn"
                title="Logout"
                onClick={triggerLogout}
              >
                <LogOut />
              </button>
            </div>
          )}
          <a
            href="https://github.com/lynxkite/lynxkite-2000"
            target="_blank"
            rel="noopener noreferrer"
            className="sidebar-nav-item"
            title="GitHub"
          >
            <BrandGithub />
            <span className="sidebar-label">GitHub</span>
          </a>
        </div>
      </div>

      <div className="management-main">
        <div className="logo">
          <a href="https://lynxkite.com/">
            <img src={logo} className="logo-image" alt="LynxKite logo" />
          </a>
          <img src={logoSparky} className="logo-image-sparky" alt="LynxKite logo" />
          <div className="tagline">The Complete Graph Data Science Platform</div>
        </div>
        <div className="management-page-content">{props.children}</div>
      </div>
    </div>
  );
}
