// Dialog for editing a folder's settings.yaml (ACL, LIM resources, ...) via an
// auto-generated form (react-jsonschema-form). The schema and current values
// come from the backend; see the "Folder settings editor" plan for the
// GET/POST /api/settings and GET /api/settings/schema endpoints this expects.
import Form from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import validator from "@rjsf/validator-ajv8";
import { forwardRef, useImperativeHandle, useRef, useState } from "react";
import useSWR from "swr";
import { apiFetch, apiJson } from "./common.ts";
import { folderSettingsTemplates } from "./FolderSettingsTemplates.tsx";

export interface AclSettings {
  read: string[];
  write: string[];
}

// `lim` maps a box name to its resource overrides (e.g. { memory: "8Gi", cpu: "2" }).
// Other top-level keys are preserved by the backend even though this type doesn't
// know about them yet.
export interface FolderSettings {
  acl: AclSettings;
  lim: Record<string, Record<string, string>>;
  [key: string]: unknown;
}

export interface FolderSettingsModalHandle {
  open: (path: string) => void;
  close: () => void;
}

// TODO(backend): remove this mock fallback once GET /api/settings/schema and
// GET /api/settings exist. It's here only so the form can be demoed before the
// backend endpoints are implemented.
const MOCK_SCHEMA: RJSFSchema = {
  type: "object",
  properties: {
    acl: {
      title: "Access control",
      type: "object",
      properties: {
        read: {
          title: "Read access",
          description: '"*" for any authenticated user, or sub:<id> / group:<name>',
          type: "array",
          items: { type: "string" },
        },
        write: {
          title: "Write access",
          description: '"*" for any authenticated user, or sub:<id> / group:<name>',
          type: "array",
          items: { type: "string" },
        },
      },
    },
    lim: {
      title: "LIM resources",
      description: "Resource overrides per box name",
      type: "object",
      additionalProperties: {
        type: "object",
        additionalProperties: { type: "string" },
      },
    },
  },
};

const MOCK_SETTINGS: FolderSettings = {
  acl: { read: ["*"], write: ["group:Eng"] },
  lim: { "Box A": { memory: "8Gi", cpu: "2" } },
};

const schemaFetcher = (url: string) => apiJson<RJSFSchema>(url).catch(() => MOCK_SCHEMA);
const settingsFetcher = (url: string) => apiJson<FolderSettings>(url).catch(() => MOCK_SETTINGS);

export const FolderSettingsModal = forwardRef<FolderSettingsModalHandle>((_props, ref) => {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const [path, setPath] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const schema = useSWR(path !== null ? "/api/settings/schema" : null, schemaFetcher);
  const settings = useSWR(
    path !== null ? `/api/settings?path=${encodeURIComponent(path)}` : null,
    settingsFetcher,
  );
  const { mutate: mutateSettings } = settings;
  const usingMockData = schema.data === MOCK_SCHEMA || settings.data === MOCK_SETTINGS;

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

  async function handleSubmit(data: { formData?: FolderSettings }) {
    if (!data.formData || path === null) return;
    setSaving(true);
    setError(null);
    try {
      const res = await apiFetch("/api/settings", {
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
      if (usingMockData) {
        // No backend to actually save to yet — just demo the round trip.
        dialogRef.current?.close();
        setPath(null);
      } else {
        setError("Failed to save settings.");
      }
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
        {usingMockData && (
          <p className="description" style={{ color: "oklch(60% 0.15 55)" }}>
            Demo data — the backend endpoints for this form aren't implemented yet.
          </p>
        )}

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
