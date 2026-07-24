import type React from "react";
import { useEffect, useState } from "react";
import { type BaseChip, CHIP_REGISTRY, getChipClass, normalizeChipType } from "./Chips.tsx";

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
  const getInitialType = () => normalizeChipType(initialChip?.type) ?? CHIP_REGISTRY[0].type;
  const [formType, setFormType] = useState(getInitialType);
  const [formData, setFormData] = useState<Record<string, string>>({});

  const ChipType = getChipClass(formType);
  const attrs = ChipType.target === "edge" ? edgeAttrs : nodeAttrs;
  const rawItems = ChipType.target === "edge" ? rawElements.edges : rawElements.nodes;

  useEffect(() => {
    setFormType(getInitialType());
  }, [initialChip]);

  useEffect(() => {
    if (initialChip && normalizeChipType(initialChip.type) === formType) {
      setFormData(initialChip.getFormData());
      return;
    }

    const emptyData: Record<string, string> = {};
    ChipType.formFields.forEach((field) => {
      emptyData[field.key] = "";
    });
    setFormData(ChipType.initFormData?.(emptyData) ?? emptyData);
  }, [formType, initialChip, ChipType]);

  const handleFieldChange = (key: string, value: string) => {
    const shouldSeedDefaults = key === "attribute" || ChipType.formFields.length === 1;

    setFormData((prev) => {
      const nextData = { ...prev, [key]: value };

      if (shouldSeedDefaults) {
        const defaultData = ChipType.getInitialData(value, rawItems, nextData);
        return { ...nextData, ...defaultData };
      }

      return nextData;
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(new ChipType(formData, initialChip?.disabled));
  };

  const isFormInvalid = ChipType.formFields.some((field) => {
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

      {ChipType.renderFormExtra?.({ formData, setFormData })}

      {ChipType.formFields.map((field) => {
        const fieldLabel = ChipType.getFormFieldLabel?.(field, formData) ?? field.label;
        return (
          <div key={field.key} style={{ display: "flex", alignItems: "center", gap: 6 }}>
            {fieldLabel && (
              <span style={{ fontSize: 11, color: "#64748b", fontWeight: 600 }}>{fieldLabel}</span>
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
                {attrs.map((attr) => (
                  <option key={attr} value={attr}>
                    {attr}
                  </option>
                ))}
              </select>
            )}
          </div>
        );
      })}

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
