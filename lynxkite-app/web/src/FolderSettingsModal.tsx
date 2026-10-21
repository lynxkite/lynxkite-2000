// Dialog for editing a folder's settings.yaml (ACL, LIM resources, ...) via an
// auto-generated form (react-jsonschema-form). The schema and current values
// come from the backend; see the "Folder settings editor" plan for the
// GET/POST /api/settings and GET /api/settings/schema endpoints this expects.
import Form from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";
import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import useSWR from "swr";
import { apiFetch, pathFetcher } from "./common.ts";
import { folderSettingsTemplates } from "./FolderSettingsTemplates.tsx";

export interface AclSettings {
  read: string[];
  write: string[];
}

export interface FolderSettingsModalHandle {
  open: (path: string) => void;
  close: () => void;
}

export const FolderSettingsModal = forwardRef<FolderSettingsModalHandle>((_props, ref) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [path, setPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const schema = useSWR(path !== null ? "/api/get_settings_schema" : null, pathFetcher);
  const settings = useSWR(
    path !== null ? `/api/dir/get_settings?path=${encodeURIComponent(path)}` : null,
    pathFetcher,
  );
  const { mutate: mutateSettings } = settings;

  useImperativeHandle(ref, () => ({
    open: (newPath: string) => {
      setPath(newPath);
      setError(null);
      dialogRef.current?.showModal();
    },
    close: () => {
      dialogRef.current?.close();
      setPath(null);
    },
  }));

  function handleClose() {
    dialogRef.current?.close();
    setPath(null);
    setError(null);
  }

  async function handleSubmit(data: { formData?: object }) {
    if (!data.formData || path === null) return;
    setSaving(true);
    setError(null);
    try {
      const res = await apiFetch("/api/dir/set_settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path, settings: data.formData }),
      });
      if (!res.ok) {
        throw new Error(`Request failed: ${res.status}`);
      }
      // Refresh the cache so reopening the dialog shows what was just saved.
      await mutateSettings(data.formData, { revalidate: true });
      dialogRef.current?.close();
      setPath(null);
    } catch (_e) {
      setError("Failed to save settings.");
    } finally {
      setSaving(false);
    }
  }

  const loading = schema.isLoading || settings.isLoading;
  const loadError = schema.error || settings.error;

  return (
    <dialog className="modal" ref={dialogRef}>
      <div className="modal-box folder-settings-modal-box w-11/12">
        <h1 className="title">Folder settings</h1>
        <p className="description">{path || "/"}</p>

        {loading && <p>Loading...</p>}
        {loadError && <p className="error">Failed to load folder settings.</p>}

        {!loading && !loadError && schema.data && settings.data && (
          <div>
            <Form
              schema={schema.data}
              formData={settings.data}
              validator={validator}
              templates={folderSettingsTemplates}
              onSubmit={handleSubmit}
            >
              {error && <p className="text-error">{error}</p>}
              <div className="modal-action">
                <button type="button" className="btn btn-ghost" onClick={handleClose}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={saving}>
                  {saving ? "Saving..." : "Save"}
                </button>
              </div>
            </Form>
          </div>
        )}

        {!loading && loadError && (
          <div className="modal-action">
            <button type="button" className="btn btn-ghost" onClick={handleClose}>
              Close
            </button>
          </div>
        )}
      </div>
    </dialog>
  );
});
