import { useEffect, useState } from "react";
import { WebsocketProvider } from "y-websocket";
import * as Y from "yjs";
import { getWebSocketParams } from "../common.ts";
import { parseProgressWorkspace } from "../progress";

export function useWorkspaceProgress(roomName: string | undefined, enabled = true) {
  const [workspaceProgress, setWorkspaceProgress] = useState<any | null>(null);

  useEffect(() => {
    if (!enabled || !roomName) {
      setWorkspaceProgress(null);
      return;
    }
    const currentRoomName = roomName;

    const doc = new Y.Doc();
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    // This workspace's own progress room. Needs only read access on the workspace.
    const room = currentRoomName
      .split("/")
      .map((segment) => encodeURIComponent(segment))
      .join("/");
    const provider = new WebsocketProvider(
      `${proto}//${location.host}/ws/progress/crdt`,
      room,
      doc,
      { connect: false },
    );
    let destroyed = false;
    getWebSocketParams().then((params) => {
      if (destroyed) return;
      provider.params = params;
      provider.connect();
    });
    const wsMap = doc.getMap("workspaces");

    function syncWorkspaceProgress() {
      const parsed = parseProgressWorkspace(wsMap.get(currentRoomName));
      setWorkspaceProgress(parsed);
    }

    wsMap.observe(syncWorkspaceProgress);
    provider.on("sync", syncWorkspaceProgress);
    provider.on("status", (event: any) => {
      if (event?.status === "connected") {
        syncWorkspaceProgress();
      }
    });

    return () => {
      destroyed = true;
      wsMap.unobserve(syncWorkspaceProgress);
      provider.destroy();
      doc.destroy();
    };
  }, [enabled, roomName]);

  return workspaceProgress;
}
