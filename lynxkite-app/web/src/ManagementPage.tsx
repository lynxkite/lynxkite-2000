// Pages like the directory browser use this to get a uniform layout.
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";
import UserMenu from "./UserMenu";

export default function ManagementPage(props: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`management-page ${props.className || ""}`}>
      <UserMenu />
      <div className="logo">
        <a href="https://lynxkite.com/">
          <img src={logo} className="logo-image" alt="LynxKite logo" />
        </a>
        <img src={logoSparky} className="logo-image-sparky" alt="LynxKite logo" />
        <div className="tagline">Discover Drugs, Graphs and the World!</div>
      </div>
      <div className="management-page-content">{props.children}</div>
    </div>
  );
}
