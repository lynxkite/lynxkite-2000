import { useState } from "react";
import ExternalLink from "~icons/tabler/external-link.jsx";
import InfoCircle from "~icons/tabler/info-circle.jsx";
import Close from "~icons/tabler/x.jsx";
import "./LegacyWorkspaceNotice.css";

export default function LegacyWorkspaceNotice({ path }: { path: string }) {
  const [dismissed, setDismissed] = useState(false);
  if (dismissed || !path.endsWith(".lynxkite.json")) return null;

  return (
    <aside
      className="legacy-workspace-alert"
      role="status"
      aria-labelledby="legacy-workspace-alert-title"
    >
      <InfoCircle className="legacy-workspace-alert-icon" aria-hidden="true" />
      <div className="legacy-workspace-alert-copy">
        <h2 id="legacy-workspace-alert-title" className="legacy-workspace-alert-title">
          Workspace format changed
        </h2>
        <p className="legacy-workspace-alert-description">
          LynxKite was updated recently. Migrate this file before opening it as a workspace.
        </p>
      </div>
      <a
        className="legacy-workspace-alert-link"
        href="https://github.com/lynxkite/lynxkite-2000/blob/main/scripts/migrate_workspaces.py"
        target="_blank"
        rel="noopener noreferrer"
      >
        View conversion script
        <ExternalLink aria-hidden="true" />
      </a>
      <button
        className="legacy-workspace-alert-close"
        type="button"
        aria-label="Dismiss workspace migration notice"
        onClick={() => setDismissed(true)}
      >
        <Close aria-hidden="true" />
      </button>
    </aside>
  );
}
