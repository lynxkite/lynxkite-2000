import NodeWithParams from "./NodeWithParams";

const NodeWithImage = (props: any) => {
  return (
    <NodeWithParams {...props}>
      {props.data.display && <img src={props.data.display} alt="Node Display" />}
    </NodeWithParams>
  );
};

export default NodeWithImage;
