import { useEffect, useState } from "react";
import NodeParameter from "./NodeParameter";

interface SelectorType {
  name: string;
  default: string;
  type: {
    enum: string[];
  };
}

interface ParameterType {
  name: string;
  default: string;
  type: {
    type: string;
  };
}

interface GroupsType {
  [key: string]: ParameterType[];
}

interface NodeGroupParameterProps {
  meta: { selector: SelectorType; groups: GroupsType };
  value: any;
  setParam: (name: string, value: any, options?: { delay: number }) => void;
  deleteParam: (name: string, options?: { delay: number }) => void;
}

export default function NodeGroupParameter({
  meta,
  value,
  setParam,
  deleteParam,
}: NodeGroupParameterProps) {
  const selector = meta.selector;
  const groups = meta.groups;
  const [selectedValue, setSelectedValue] = useState<string>(
    value || selector.default,
  );

  const handleSelectorChange = (value: any, opts?: { delay: number }) => {
    setSelectedValue(value);
    setParam(selector.name, value, opts);
  };

  useEffect(() => {
    // Clean possible previous parameters first
    Object.values(groups).flatMap((group) =>
      group.map((entry) => deleteParam(entry.name)),
    );
    for (const param of groups[selectedValue]) {
      setParam(param.name, param.default);
    }
  }, [selectedValue]);

  return (
    <NodeParameter
      name={selector.name}
      key={selector.name}
      value={selectedValue}
      meta={selector}
      onChange={handleSelectorChange}
    />
  );
}
