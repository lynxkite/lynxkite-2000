import NodeParameter, { type UpdateOptions } from "./NodeParameter";

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
  data: any;
  setParam: (name: string, value: any, options: UpdateOptions) => void;
}

export default function NodeGroupParameter({ meta, data, setParam }: NodeGroupParameterProps) {
  const selector = meta.selector;
  const selectorValue = data.params[selector.name] || selector.default;
  const group = meta.groups[selectorValue] || [];

  return (
    <>
      {group.map((meta: any) => (
        <NodeParameter
          name={meta.name}
          key={meta.name}
          value={data.params[meta.name] ?? meta.default}
          data={data}
          meta={meta}
          setParam={setParam}
        />
      ))}
    </>
  );
}
