import type React from "react";
import { useEffect, useState } from "react";
import { type BaseChip, CHIP_REGISTRY, getChipClass } from "./Chips.tsx";

interface ChipFormProps {
  nodeAttrs: string[];
  edgeAttrs: string[];
  initialChip: BaseChip | null;
  onSubmit: (newChip: BaseChip) => void;
  rawElements: { nodes: any[]; edges: any[] };
}

const THEME = {
  button: { bg: "rgb(33 168 96 / 0.78)", text: "#ffffff", disabledBg: "#cbd5e1" },
  border: "#e2e8f0",
};

export default function ChipForm({
  nodeAttrs,
  edgeAttrs,
  initialChip,
  onSubmit,
  rawElements,
}: ChipFormProps) {
  const [formType, setFormType] = useState(initialChip?.type || CHIP_REGISTRY[0].type);
  const [formData, setFormData] = useState<Record<string, string>>({});

  const ActiveClass = getChipClass(formType);
  const targetAttrs = ActiveClass.target === "edge" ? edgeAttrs : nodeAttrs;
  const activeRawItems = ActiveClass.target === "edge" ? rawElements.edges : rawElements.nodes;

  useEffect(() => {
    const isEditingCurrentType = initialChip && initialChip.type === formType;
    if (isEditingCurrentType) {
      setFormData(initialChip.getFormData());
      return;
    }

    const emptyData: Record<string, string> = {};
    ActiveClass.formFields.forEach((field) => {
      emptyData[field.key] = "";
    });
    setFormData(emptyData);
  }, [formType, initialChip, ActiveClass]);

  const handleFieldChange = (key: string, value: string) => {
    const shouldSeedDefaults = key === "attribute" || ActiveClass.formFields.length === 1;

    setFormData((prev) => {
      const nextData = { ...prev, [key]: value };

      if (shouldSeedDefaults) {
        const defaultData = ActiveClass.getInitialData(value, activeRawItems, nextData);
        return { ...nextData, ...defaultData };
      }

      return nextData;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(new ActiveClass(formData, initialChip?.disabled));
  };

  const isFormInvalid = ActiveClass.formFields.some((field) => {
    const val = formData[field.key];
    return !val || val === "";
  });

  const selectStyle = {
    padding: "4px 8px",
    borderRadius: 6,
    border: `1px solid ${THEME.border}`,
    outline: "none",
    backgroundColor: "#fff",
    fontSize: 12,
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: "#fff",
        padding: "6px 12px",
        borderRadius: 10,
        boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
        border: `1px solid ${THEME.border}`,
      }}
    >
      <select value={formType} onChange={(e) => setFormType(e.target.value)} style={selectStyle}>
        {CHIP_REGISTRY.map((c) => (
          <option key={c.type} value={c.type}>
            {c.displayName}
          </option>
        ))}
      </select>

      {ActiveClass.formFields.map((field) => (
        <div key={field.key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
          {field.label && (
            <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}>{field.label}</span>
          )}
          {field.type === "number" ? (
            <input
              type="number"
              min={1}
              value={formData[field.key] ?? ""}
              onChange={(e) => setFormData({ ...formData, [field.key]: e.target.value })}
              style={{ ...selectStyle, width: 60 }}
            />
          ) : (
            <select
              value={formData[field.key] || ""}
              onChange={(e) => handleFieldChange(field.key, e.target.value)}
              style={selectStyle}
            >
              <option value=""></option>
              {targetAttrs.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          )}
        </div>
      ))}

      <button
        type="submit"
        disabled={isFormInvalid}
        style={{
          padding: "4px 12px",
          fontSize: 12,
          background: isFormInvalid ? THEME.button.disabledBg : THEME.button.bg,
          color: THEME.button.text,
          border: "none",
          borderRadius: 6,
          cursor: isFormInvalid ? "not-allowed" : "pointer",
          fontWeight: "bold",
        }}
      >
        {initialChip ? "Save" : "Apply"}
      </button>
    </form>
  );
}
