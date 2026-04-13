// A system-wide progress page, that gives an overview of workspaces running, resources used, etc.

import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import ScaleDown from "~icons/tabler/arrow-down";
import Back from "~icons/tabler/arrow-left";
import ScaleUp from "~icons/tabler/arrow-up";
import Edit from "~icons/tabler/edit";
import Pause from "~icons/tabler/player-pause-filled";
import Play from "~icons/tabler/player-play-filled";
import Stop from "~icons/tabler/player-stop-filled";
import UserFilled from "~icons/tabler/user-filled";
import logo from "./assets/logo.png";
import logoSparky from "./assets/logo-sparky.jpg";

const echarts = await import("echarts");

// Generate fake per-day GPU-hours for a user over the last 30 days.
function generateDailyUsage(avgHours: number): number[] {
  const days = [];
  for (let i = 0; i < 30; i++) {
    days.push(
      Math.max(0, Math.round(avgHours + (Math.sin(i * 1.3) + Math.cos(i * 0.7)) * avgHours * 0.4)),
    );
  }
  return days;
}

function timeLeft(ws: any): string {
  if (ws.eta_seconds == null) return "";
  if (ws.eta_seconds <= 0) return "done";
  const minutes = Math.floor(ws.eta_seconds / 60);
  const seconds = Math.floor(ws.eta_seconds % 60);
  if (minutes > 0) return `~${minutes}m ${seconds}s left`;
  return `~${seconds}s left`;
}

export default function ProgressPage() {
  // Update every second so we see the time left count down.
  const [, setTick] = useState(0);
  useEffect(() => {
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, []);
  const [currentTab, setCurrentTab] = useState("workspaces");
  const tabs = [
    { id: "workspaces", label: "Running workspaces" },
    { id: "gpu-services", label: "Running GPU services" },
    { id: "users", label: "Users & groups" },
  ];

  // Mock data for now.
  const [data, setData] = useState<{
    workspaces: any[];
    gpuServices: any[];
    users: any[];
  }>({
    workspaces: [],
    gpuServices: [],
    users: [
      {
        name: "Botond Banhidi",
        group: "Engineering",
        gpuQuota: 500,
        dailyUsage: generateDailyUsage(15),
        email: "botond.banhidi@lynxkite.com",
      },
      {
        name: "Daniel Darabos",
        group: "Engineering",
        gpuQuota: 800,
        dailyUsage: generateDailyUsage(22),
        email: "daniel.darabos@lynxkite.com",
      },
      {
        name: "Derek Smith",
        group: "Drug Discovery",
        gpuQuota: 2000,
        dailyUsage: generateDailyUsage(55),
        email: "derek.smith@lynxkite.com",
      },
      {
        name: "Dora Gera",
        group: "Drug Discovery",
        gpuQuota: 1500,
        dailyUsage: generateDailyUsage(40),
        email: "dora.gera@lynxkite.com",
      },
      {
        name: "Gergo Szabo",
        group: "Engineering",
        gpuQuota: 1500,
        dailyUsage: generateDailyUsage(35),
        email: "gergo.szabo@lynxkite.com",
      },
      {
        name: "Gyorgy Lajtai",
        group: "Engineering",
        gpuQuota: 2500,
        dailyUsage: generateDailyUsage(70),
        email: "gyorgy.lajtai@lynxkite.com",
      },
      {
        name: "Livia Babos",
        group: "Micro RNA",
        gpuQuota: 800,
        dailyUsage: generateDailyUsage(15),
        email: "livia.babos@lynxkite.com",
      },
      {
        name: "Rajat Kumar Pal",
        group: "Molecular Simulation",
        gpuQuota: 3000,
        dailyUsage: generateDailyUsage(80),
        email: "rajat.kumar.pal@lynxkite.com",
      },
    ],
  });

  async function pauseWorkspace(roomName: string, paused: boolean) {
    try {
      await fetch("/api/pause_workspace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room_name: roomName, paused }),
      });
    } catch (e) {
      console.error("Failed to pause workspace", e);
    }
  }

  async function stopWorkspace(roomName: string) {
    try {
      await fetch("/api/stop_workspace", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ room_name: roomName }),
      });
    } catch (e) {
      console.error("Failed to stop workspace", e);
    }
  }

  async function scaleGpuService(gpuService: any, newReplicaCount: number) {
    const targetReplicas = Math.max(0, newReplicaCount);
    try {
      await fetch("/api/scale_gpu_service", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: gpuService.name,
          namespace: gpuService.publisher,
          replicas: targetReplicas,
        }),
      });
    } catch (e) {
      console.error("Failed to scale GPU service", e);
    }
  }
  useEffect(() => {
    const doc = new Y.Doc();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const provider = new WebsocketProvider(
      `${proto}//${location.host}/ws/progress/crdt`,
      "progress",
      doc,
    );

    const wsMap = doc.getMap("workspaces");
    const gpuServicesText = doc.getText("gpu_services");

    function parseWorkspace(value: unknown) {
      if (typeof value === "string") {
        return JSON.parse(value);
      }
      return value as any;
    }

    function syncWorkspaces() {
      const workspaces: any[] = [];
      for (const value of (wsMap as Y.Map<unknown>).values()) {
        try {
          const ws = parseWorkspace(value);
          if (ws && typeof ws === "object") {
            workspaces.push({ ...ws, user: "Test User" });
          }
        } catch (e) {
          console.warn("failed to parse workspace entry from CRDT", e);
        }
      }
      setData((prev) => ({ ...prev, workspaces }));
    }

    function syncGpuServices() {
      const raw = gpuServicesText.toString();
      if (!raw) {
        return;
      }
      try {
        const gpuServices = JSON.parse(raw);
        if (Array.isArray(gpuServices)) {
          setData((prev) => ({ ...prev, gpuServices }));
        }
      } catch {
        // ignore malformed JSON during partial updates
      }
    }

    wsMap.observe(syncWorkspaces);
    gpuServicesText.observe(syncGpuServices);
    // Sync once the provider has connected and received initial state.
    provider.on("sync", () => {
      syncWorkspaces();
      syncGpuServices();
    });
    provider.on("status", (event: any) => {
      if (event?.status === "connected") {
        syncWorkspaces();
        syncGpuServices();
      }
    });

    return () => {
      wsMap.unobserve(syncWorkspaces);
      gpuServicesText.unobserve(syncGpuServices);
      provider.destroy();
      doc.destroy();
    };
  }, []);

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
        <div role="tablist" className="tabs tabs-border">
          <div className="tab">
            <Link to="/">
              <Back />
            </Link>
          </div>
          {tabs.map((tab) => (
            <a
              key={tab.id}
              role="tab"
              onClick={() => setCurrentTab(tab.id)}
              className={currentTab === tab.id ? "tab tab-active" : "tab"}
            >
              {tab.label}
            </a>
          ))}
        </div>

        {currentTab === "workspaces" && (
          <Workspaces
            workspaces={data.workspaces}
            onPause={(roomName, paused) => pauseWorkspace(roomName, paused)}
            onStop={(roomName) => stopWorkspace(roomName)}
          />
        )}
        {currentTab === "gpu-services" && (
          <GpuServices gpuServices={data.gpuServices} scaleGpuService={scaleGpuService} />
        )}
        {currentTab === "users" && <UsersAndGroups users={data.users} />}
      </div>
    </div>
  );
}

function Workspaces(props: {
  workspaces: any[];
  onPause: (roomName: string, paused: boolean) => void;
  onStop: (roomName: string) => void;
}) {
  if ((props.workspaces?.length ?? 0) === 0) {
    return <div>No workspaces in progress.</div>;
  }
  return (
    <table className="progress-table">
      <thead>
        <tr>
          <th>Workspace</th>
          <th>User</th>
          <th colSpan={2}>Progress</th>
          <th>GPUs</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {props.workspaces.map((ws) => {
          const roomName = ws.room_name || ws.name;
          const boxFraction = ws.boxes_total > 0 ? ws.boxes_done / ws.boxes_total : 0;
          const tqdm = ws.active_node?.tqdm;
          // Combined progress: box fraction covers full bar, tqdm refines the current box's slice
          const tqdmFraction = tqdm?.total > 0 ? tqdm.n / tqdm.total : null;
          // Each box is 1/total wide. The active box contributes its tqdm progress within that slice.
          const combinedProgress =
            tqdmFraction != null ? (ws.boxes_done + tqdmFraction) / ws.boxes_total : boxFraction;
          return (
            <tr key={ws.name}>
              <td className="workspace-name">{ws.name}</td>
              <td className="workspace-user">{ws.user}</td>
              <td className="workspace-progress">
                <div className="progress-details">
                  {ws.active_node && (
                    <span className="active-node-label">
                      {ws.active_node.title}
                      {tqdm?.total != null && ` (${tqdm.n}/${tqdm.total})`}
                    </span>
                  )}
                  <progress
                    className={`progress progress-${ws.status === "active" ? "primary" : "neutral"} w-50`}
                    value={combinedProgress * 100}
                    max={100}
                  />
                </div>
              </td>
              <td className="workspace-eta">
                {ws.boxes_total > 0 && `${ws.boxes_done}/${ws.boxes_total}`}
                {timeLeft(ws) && <span> {timeLeft(ws)}</span>}
              </td>
              <td className="workspace-resources">{ws.gpus || "—"}</td>
              <td className="table-actions">
                {ws.paused ? (
                  <button
                    className="btn btn-sm"
                    title="Resume"
                    onClick={() => props.onPause(roomName, false)}
                  >
                    <Play />
                  </button>
                ) : (
                  <button
                    className="btn btn-sm"
                    title="Pause"
                    onClick={() => props.onPause(roomName, true)}
                  >
                    <Pause />
                  </button>
                )}
                <button
                  className="btn btn-sm"
                  title="Stop (reset all boxes)"
                  onClick={() => props.onStop(roomName)}
                >
                  <Stop />
                </button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function GpuServices(props: {
  gpuServices: any[];
  scaleGpuService: (gpuService: any, replicas: number) => void;
}) {
  const [showUsers, setShowUsers] = useState({} as Record<string, boolean>);
  if ((props.gpuServices?.length ?? 0) === 0) {
    return <div>No GPU services deployed.</div>;
  }
  return (
    <table className="progress-table">
      <thead>
        <tr>
          <th>GPU service</th>
          <th>Replicas</th>
          <th>Status</th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        {props.gpuServices.map((gpuService) => (
          <tr key={gpuService.name}>
            <td
              className="gpu-service-name"
              onClick={() =>
                setShowUsers((prev) => ({ ...prev, [gpuService.name]: !prev[gpuService.name] }))
              }
            >
              {gpuService.name}{" "}
              <span className="gpu-service-publisher">from {gpuService.publisher}</span>
              {showUsers[gpuService.name] && (
                <div className="gpu-service-users">
                  Used by workspaces:
                  {gpuService.usedByWorkspaces.map((ws: string) => (
                    <div className="gpu-service-user" key={ws}>
                      {ws}
                    </div>
                  ))}
                </div>
              )}
            </td>
            <td className="gpu-service-replica-count" title="Running / set">
              {gpuService.replicasHealthy} / {gpuService.replicasRequested}
            </td>
            <td className="gpu-service-status">
              {" "}
              {gpuService.replicasHealthy !== gpuService.replicasRequested
                ? "resizing"
                : gpuService.status}{" "}
            </td>
            <td className="table-actions">
              <button
                className="btn btn-sm"
                title="Scale up"
                onClick={() => props.scaleGpuService(gpuService, gpuService.replicasRequested + 1)}
              >
                <ScaleUp />
              </button>
              <button
                className="btn btn-sm"
                title="Scale down"
                onClick={() => props.scaleGpuService(gpuService, gpuService.replicasRequested - 1)}
              >
                <ScaleDown />
              </button>
              <button
                className="btn btn-sm"
                title="Stop"
                onClick={() => props.scaleGpuService(gpuService, 0)}
              >
                <Stop />
              </button>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function UserUsageChart(props: { dailyUsage: number[]; gpuQuota: number }) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts>(null);
  useEffect(() => {
    if (!chartRef.current) return;
    chartInstance.current = echarts.init(chartRef.current, null, { renderer: "canvas" });
    const today = new Date();
    const labels = props.dailyUsage.map((_, i) => {
      const d = new Date(today);
      d.setDate(d.getDate() - 29 + i);
      return `${d.getMonth() + 1}/${d.getDate()}`;
    });
    const dailyQuota = Math.round(props.gpuQuota / 30);
    chartInstance.current.setOption({
      tooltip: { trigger: "axis" },
      xAxis: { type: "category", data: labels, axisLabel: { rotate: 45 } },
      yAxis: { type: "value", name: "GPU-hours" },
      series: [
        {
          name: "Usage",
          type: "bar",
          data: props.dailyUsage,
          itemStyle: { color: "oklch(75% 0.2 230)" },
        },
        {
          name: "Daily quota",
          type: "line",
          data: Array(30).fill(dailyQuota),
          lineStyle: { type: "dashed", color: "oklch(75% 0.2 55)" },
          symbol: "none",
        },
      ],
    });
    return () => {
      chartInstance.current?.dispose();
    };
  }, [props.dailyUsage, props.gpuQuota]);
  return <div ref={chartRef} style={{ width: "100%", height: 220 }} />;
}

const ALL_GROUPS = [
  "Engineering",
  "Drug Discovery",
  "Micro RNA",
  "Molecular Simulation",
  "Management",
  "Data Science",
];
const ALL_GPU_TYPES = ["H100", "H200", "GB200", "A100", "L40S", "B200"];

function PolicyEditDialog(props: { policyName: string; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [limitMode, setLimitMode] = useState<"monthly" | "weekly" | "daily">("monthly");
  const [limitHours, setLimitHours] = useState(1000);
  const [gpuTypes, setGpuTypes] = useState<string[]>(["H100", "H200", "GB200"]);
  const [offHoursEnabled, setOffHoursEnabled] = useState(false);
  const [offHoursFrom, setOffHoursFrom] = useState("17:00");
  const [offHoursTo, setOffHoursTo] = useState("06:00");
  const [priority, setPriority] = useState("medium");
  const [preemptible, setPreemptible] = useState(false);
  const [quantum, setQuantum] = useState(false);
  useEffect(() => {
    dialogRef.current?.showModal();
  }, []);
  const availableGpuTypes = ALL_GPU_TYPES.filter((t) => !gpuTypes.includes(t));
  return (
    <dialog
      ref={dialogRef}
      className="modal"
      style={{ zIndex: 1100 }}
      onClose={(e) => {
        e.stopPropagation();
        props.onClose();
      }}
    >
      <div className="modal-box">
        <h3 className="font-bold text-lg">{props.policyName}</h3>
        <label className="label">GPU hour limit</label>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <input
            className="input input-bordered w-32"
            type="number"
            value={limitHours}
            onChange={(e) => setLimitHours(Number(e.target.value))}
          />
          <span>hours per</span>
          <select
            className="select select-bordered"
            value={limitMode}
            onChange={(e) => setLimitMode(e.target.value as "monthly" | "weekly" | "daily")}
          >
            <option value="daily">day</option>
            <option value="weekly">week</option>
            <option value="monthly">month</option>
          </select>
        </div>
        <label className="label">Priority</label>
        <select
          className="select select-bordered w-full"
          value={priority}
          onChange={(e) => setPriority(e.target.value)}
        >
          <option value="lowest">Lowest</option>
          <option value="low">Low</option>
          <option value="medium">Medium</option>
          <option value="high">High</option>
          <option value="highest">Highest</option>
        </select>
        <label className="label">Allowed GPU types</label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          {gpuTypes.map((t) => (
            <span key={t} className="badge badge-primary gap-1">
              {t}
              <button
                className="btn btn-ghost btn-xs px-0"
                onClick={() => setGpuTypes(gpuTypes.filter((x) => x !== t))}
              >
                ✕
              </button>
            </span>
          ))}
          {availableGpuTypes.length > 0 && (
            <div className="dropdown">
              <button type="button" tabIndex={0} className="btn btn-xs btn-circle btn-outline">
                +
              </button>
              <ul
                tabIndex={0}
                className="dropdown-content menu bg-base-100 rounded-box z-10 w-52 p-2 shadow-sm"
              >
                {availableGpuTypes.map((t) => (
                  <li key={t}>
                    <a onClick={() => setGpuTypes([...gpuTypes, t])}>{t}</a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <div style={{ display: "flex", gap: 16, marginTop: 8 }}>
          <label className="label cursor-pointer gap-2">
            <input
              type="checkbox"
              className="checkbox checkbox-sm"
              checked={preemptible}
              onChange={(e) => setPreemptible(e.target.checked)}
            />
            Allow pre-emptible instances
          </label>
          <label className="label cursor-pointer gap-2">
            <input
              type="checkbox"
              className="checkbox checkbox-sm"
              checked={quantum}
              onChange={(e) => setQuantum(e.target.checked)}
            />
            Allow quantum computing instances
          </label>
        </div>
        <label className="label">Off hours</label>
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <label className="label cursor-pointer gap-2">
            <input
              type="checkbox"
              className="checkbox checkbox-sm"
              checked={offHoursEnabled}
              onChange={(e) => setOffHoursEnabled(e.target.checked)}
            />
            Automatically shut down jobs from
          </label>
          <input
            type="time"
            className="input input-bordered input-sm w-28"
            value={offHoursFrom}
            disabled={!offHoursEnabled}
            onChange={(e) => setOffHoursFrom(e.target.value)}
          />
          <span>to</span>
          <input
            type="time"
            className="input input-bordered input-sm w-28"
            value={offHoursTo}
            disabled={!offHoursEnabled}
            onChange={(e) => setOffHoursTo(e.target.value)}
          />
        </div>
        <div className="modal-action">
          <form method="dialog">
            <button className="btn">Close</button>
          </form>
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  );
}

function UserEditDialog(props: { user: any; onClose: () => void }) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [name, setName] = useState(props.user.name);
  const [email, setEmail] = useState(props.user.email ?? "");
  const [groups, setGroups] = useState<string[]>(
    Array.isArray(props.user.group) ? props.user.group : [props.user.group],
  );
  const [editingPolicy, setEditingPolicy] = useState<string | null>(null);
  useEffect(() => {
    dialogRef.current?.showModal();
  }, []);
  const availableGroups = ALL_GROUPS.filter((g) => !groups.includes(g));
  return (
    <dialog ref={dialogRef} className="modal" onClose={props.onClose}>
      <div className="modal-box">
        <h3 className="font-bold text-lg">Edit User</h3>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 8,
            marginBottom: 16,
          }}
        >
          <div style={{ fontSize: 64, color: "oklch(60% 0.15 230)" }}>
            <UserFilled />
          </div>
          <a href="#" className="text-sm">
            <Edit /> Change profile picture
          </a>
        </div>
        <label className="label">Name</label>
        <input
          className="input input-bordered w-full"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <label className="label">Email</label>
        <input
          className="input input-bordered w-full"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <div style={{ marginTop: 16 }}>
          <button
            className="btn btn-warning btn-sm"
            onClick={() => alert("Password reset link sent.")}
          >
            Reset password
          </button>
        </div>
        <label className="label">Groups</label>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
          {groups.map((g) => (
            <span key={g} className="badge badge-primary gap-1">
              {g}
              <button
                className="btn btn-ghost btn-xs px-0"
                onClick={() => setGroups(groups.filter((x) => x !== g))}
              >
                ✕
              </button>
            </span>
          ))}
          {availableGroups.length > 0 && (
            <div className="dropdown">
              <button type="button" tabIndex={0} className="btn btn-xs btn-circle btn-outline">
                +
              </button>
              <ul
                tabIndex={0}
                className="dropdown-content menu bg-base-100 rounded-box z-10 w-52 p-2 shadow-sm"
              >
                {availableGroups.map((g) => (
                  <li key={g}>
                    <a onClick={() => setGroups([...groups, g])}>{g}</a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <label className="label">Applied policies</label>
        <ul className="list-disc list-inside">
          <li>
            {name} user policy{" "}
            <button
              className="btn btn-ghost btn-xs"
              onClick={() => setEditingPolicy(`${name} user policy`)}
            >
              <Edit />
            </button>
          </li>
          {groups.map((g) => (
            <li key={g}>
              {g} group policy{" "}
              <button
                className="btn btn-ghost btn-xs"
                onClick={() => setEditingPolicy(`${g} group policy`)}
              >
                <Edit />
              </button>
            </li>
          ))}
        </ul>
        {editingPolicy && (
          <PolicyEditDialog policyName={editingPolicy} onClose={() => setEditingPolicy(null)} />
        )}
        <div className="modal-action">
          <form method="dialog">
            <button className="btn">Close</button>
          </form>
        </div>
      </div>
      <form method="dialog" className="modal-backdrop">
        <button>close</button>
      </form>
    </dialog>
  );
}

function UsersAndGroups(props: { users: any[] }) {
  const [expandedUser, setExpandedUser] = useState<string | null>(null);
  const [editingUser, setEditingUser] = useState<any | null>(null);
  if ((props.users?.length ?? 0) === 0) {
    return <div>No users.</div>;
  }
  return (
    <>
      <table className="progress-table">
        <thead>
          <tr>
            <th>User</th>
            <th>Group</th>
            <th>Quota used (h)</th>
          </tr>
        </thead>
        <tbody>
          {props.users.map((user) => {
            const totalUse = user.dailyUsage.reduce((a: number, b: number) => a + b, 0);
            const expanded = expandedUser === user.name;
            return (
              <React.Fragment key={user.name}>
                <tr
                  className={expanded ? "active" : ""}
                  style={{ cursor: "pointer" }}
                  onClick={() => setExpandedUser(expanded ? null : user.name)}
                >
                  <td>{user.name}</td>
                  <td>{user.group}</td>
                  <td>
                    <progress
                      className={`progress ${totalUse > user.gpuQuota ? "progress-error" : "progress-primary"} w-50`}
                      value={totalUse}
                      max={user.gpuQuota}
                    />{" "}
                    {totalUse.toLocaleString()}/{user.gpuQuota.toLocaleString()}
                  </td>
                </tr>
                {expanded && (
                  <tr>
                    <td colSpan={3}>
                      <div style={{ display: "flex", alignItems: "start" }}>
                        <UserUsageChart dailyUsage={user.dailyUsage} gpuQuota={user.gpuQuota} />
                        <button
                          className="btn btn-lg btn-ghost"
                          title="Edit user"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEditingUser(user);
                          }}
                        >
                          <Edit />
                        </button>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            );
          })}
        </tbody>
      </table>
      {editingUser && <UserEditDialog user={editingUser} onClose={() => setEditingUser(null)} />}
    </>
  );
}
