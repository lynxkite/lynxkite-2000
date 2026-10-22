// Pages like the directory browser use this to get a uniform layout.
import { useState } from "react";
import { NavLink } from "react-router";
import Activity from "~icons/tabler/activity";
import Book from "~icons/tabler/book";
import BrandGithub from "~icons/tabler/brand-github";
import Folder from "~icons/tabler/folder";
import LogIn from "~icons/tabler/login";
import LogOut from "~icons/tabler/logout";
import Menu from "~icons/tabler/menu-2";
import SettingsIcon from "~icons/tabler/settings";
import UserIcon from "~icons/tabler/user";
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";
import { getConfig, triggerLogin, triggerLogout, useCurrentUser } from "./common.ts";
import { SidebarFavorites, SidebarRecent, SidebarSearch } from "./SidebarExtras.tsx";

const SIDEBAR_OPEN_KEY = "lynxkite-sidebar-open";

const NAV_ITEMS = [
  { to: "/dir", icon: <Folder />, label: "Files" },
  { to: "/progress", icon: <Activity />, label: "Progress" },
];

export default function ManagementPage(props: {
  children: React.ReactNode;
  className?: string;
  sidebarBottom?: React.ReactNode;
}) {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [sidebarOpen, setSidebarOpenState] = useState(() => {
    try {
      return localStorage.getItem(SIDEBAR_OPEN_KEY) !== "false";
    } catch {
      return true;
    }
  });
  // Each page renders its own ManagementPage, so the state is kept in localStorage.
  function setSidebarOpen(open: boolean) {
    setSidebarOpenState(open);
    try {
      localStorage.setItem(SIDEBAR_OPEN_KEY, String(open));
    } catch {
      // Storage may be blocked, for example when embedded in another site.
    }
  }
  const config = getConfig();
  const currentUser = useCurrentUser();
  const authEnabled = !!config.authentication_issuer;

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

        <SidebarSearch open={sidebarOpen} onExpand={() => setSidebarOpen(true)} />

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
          <button
            type="button"
            className="sidebar-nav-item"
            title="Settings"
            onClick={() => setSettingsOpen(true)}
          >
            <SettingsIcon />
            <span className="sidebar-label">Settings</span>
          </button>
        </nav>

        <div className="sidebar-inner">
          <SidebarFavorites />
          <SidebarRecent />
          {props.sidebarBottom}
        </div>

        {authEnabled && (
          <div className="sidebar-user-section">
            {authEnabled && currentUser === null && (
              <button
                type="button"
                className="sidebar-nav-item"
                title="Sign in"
                onClick={triggerLogin}
              >
                <LogIn />
                <span className="sidebar-label">Sign in</span>
              </button>
            )}
            {authEnabled && currentUser !== null && (
              // biome-ignore lint/a11y/useKeyWithClickEvents: expands the collapsed sidebar
              // biome-ignore lint/a11y/noStaticElementInteractions: expands the collapsed sidebar
              <div className="sidebar-user" onClick={() => !sidebarOpen && setSidebarOpen(true)}>
                <UserIcon className="sidebar-user-icon" />
                <span className="sidebar-label sidebar-user-name">{currentUser}</span>
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
          </div>
        )}

        <div className="sidebar-footer">
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

      {settingsOpen && (
        // biome-ignore lint/a11y/useKeyWithClickEvents: mockup dialog
        <div className="settings-popup-backdrop" onClick={() => setSettingsOpen(false)}>
          {/* biome-ignore lint/a11y/useKeyWithClickEvents: mockup dialog */}
          <div className="settings-popup" onClick={(e) => e.stopPropagation()}>
            <h3>Folder settings</h3>
            <p>Folder demos will be here</p>
            <button type="button" onClick={() => setSettingsOpen(false)}>
              Close
            </button>
          </div>
        </div>
      )}

      <div className="management-main">
        <div className="logo">
          <a href="https://lynxkite.com/">
            <img src={logo} className="logo-image" alt="LynxKite logo" />
          </a>
          <img src={logoSparky} className="logo-image-sparky" alt="LynxKite logo" />
          <div className="tagline">Discover Drugs, Graphs and the World!</div>
        </div>
        <div className="management-page-content">{props.children}</div>
      </div>
    </div>
  );
}
