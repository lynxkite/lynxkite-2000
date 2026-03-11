// An system-wide progress page, that gives an overview of which workspaces are running, what resources they use, etc.
import { Link } from "react-router";
import Home from "~icons/tabler/home";
import Pause from "~icons/tabler/player-pause-filled";
import Stop from "~icons/tabler/player-stop-filled";
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";

function timeLeft(workspace: any) {
  const now = Date.now();
  const eta = workspace.eta;
  const diff = eta - now;

  if (diff <= 0) {
    return "Done";
  }

  const minutes = Math.floor(diff / (1000 * 60));
  const seconds = Math.floor((diff % (1000 * 60)) / 1000);

  return `${minutes}m ${seconds}s left`;
}

export default function ProgressPage() {
  // Mock data for now.
  const data = {
    workspaces: [
      {
        name: "Drug discovery pipeline",
        user: "Derek Smith",
        start_time: Date.now() - 1000 * 60 * 60 * 2,
        eta: Date.now() + 1000 * 60 * 34.3,
        boxes_done: 18,
        total_boxes: 24,
        current_box_progress: 0.45,
        overall_progress: 0.75,
        resources: { gpus: 8 },
      },
      {
        name: "Generating receptor conformations",
        user: "Derek Smith",
        start_time: Date.now() - 1000 * 60 * 45,
        eta: Date.now() + 1000 * 60 * 21.1,
        boxes_done: 3,
        total_boxes: 12,
        current_box_progress: 0.6,
        overall_progress: 0.28,
        resources: { gpus: 2 },
      },
      {
        name: "Training docking model",
        user: "Dora Gera",
        start_time: Date.now() - 1000 * 60 * 10,
        eta: Date.now() + 1000 * 60 * 122.5,
        boxes_done: 1,
        total_boxes: 20,
        current_box_progress: 0.05,
        overall_progress: 0.05,
        resources: { gpus: 4 },
      },
      {
        name: "Side effects and toxicity screening",
        user: "Rajat Kumar Pal",
        start_time: Date.now() - 1000 * 60 * 60 * 6,
        eta: Date.now() + 1000 * 60 * 60 * 1.77,
        boxes_done: 40,
        total_boxes: 50,
        current_box_progress: 0.9,
        overall_progress: 0.82,
        resources: { gpus: 6 },
      },
      {
        name: "Molecular dynamics simulation",
        user: "Rajat Kumar Pal",
        start_time: Date.now() - 1000 * 60 * 60 * 24,
        eta: Date.now() + 1000 * 60 * 60 * 12.1,
        boxes_done: 120,
        total_boxes: 200,
        current_box_progress: 0.4,
        overall_progress: 0.6,
        resources: { gpus: 16 },
      },
      {
        name: "MicroRNA experiment",
        user: "Livia Babos",
        start_time: Date.now() - 1000 * 60 * 15,
        eta: Date.now() + 1000 * 60 * 45.8,
        boxes_done: 2,
        total_boxes: 8,
        current_box_progress: 0.25,
        overall_progress: 0.2,
        resources: { gpus: 1 },
      },
    ],
  };

  return (
    <div className="progress-page">
      <div className="logo">
        <a href="https://lynxkite.com/">
          <img src={logo} className="logo-image" alt="LynxKite logo" />
        </a>
        <img src={logoSparky} className="logo-image-sparky" alt="LynxKite logo" />
        <div className="tagline">The Complete Graph Data Science Platform</div>
      </div>
      <div className="entry-list">
        <div><Link to="/"><Home></Home></Link></div>
        {data.workspaces.length === 0 ? (
          <div>No workspaces in progress.</div>
        ) : (
          <table className="progress-table">
            <thead>
              <tr>
                <th>Workspace</th>
                <th>User</th>
                <th colSpan="2">Progress</th>
                <th>GPUs</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {data.workspaces.map((ws) => (
                <tr key={ws.name}>
                  <td className="workspace-name">{ws.name}</td>
                  <td className="workspace-user">{ws.user}</td>
                  <td className="workspace-progress">
                    <progress className="progress progress-primary w-50" value={ws.overall_progress * 100} max="100"></progress>
                  </td>
                  <td className="workspace-eta">
                    {timeLeft(ws)}
                  </td>
                  <td className="workspace-resources">{ws.resources.gpus}</td>
                  <td className="workspace-actions">
                    <button className="btn btn-sm"><Pause/></button>
                    <button className="btn btn-sm"><Stop/></button>
                    </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
