// DaisyUI-styled templates for react-jsonschema-form, so the folder settings
// form looks like the rest of the app instead of Bootstrap.
import {
  ADDITIONAL_PROPERTY_FLAG,
  type ArrayFieldItemTemplateProps,
  type ArrayFieldTemplateProps,
  type BaseInputTemplateProps,
  buttonId,
  canExpand,
  type DescriptionFieldProps,
  type ErrorListProps,
  type FieldErrorProps,
  type FieldHelpProps,
  type FieldTemplateProps,
  getInputProps,
  type IconButtonProps,
  type ObjectFieldTemplateProps,
  type TemplatesType,
  type TitleFieldProps,
} from "@rjsf/utils";
import type { ReactNode } from "react";
import ArrowDown from "~icons/tabler/arrow-down";
import ArrowUp from "~icons/tabler/arrow-up";
import Copy from "~icons/tabler/copy";
import Plus from "~icons/tabler/plus";
import X from "~icons/tabler/x";

function IconButton({ icon, className, title, ...props }: IconButtonProps & { icon: ReactNode }) {
  const { uiSchema: _u, registry: _r, iconType: _i, ...rest } = props;
  return (
    <button
      type="button"
      className={`btn btn-sm btn-ghost btn-square ${className ?? ""}`}
      title={title}
      {...rest}
    >
      {icon}
    </button>
  );
}

const AddButton = ({ id, onClick, disabled, className }: IconButtonProps) => (
  <button
    type="button"
    id={id}
    className={`btn btn-sm btn-outline self-start ${className ?? ""}`}
    onClick={onClick}
    disabled={disabled}
  >
    <Plus /> Add
  </button>
);

const RemoveButton = (props: IconButtonProps) => (
  <IconButton {...props} title="Remove" icon={<X />} />
);
const CopyButton = (props: IconButtonProps) => (
  <IconButton {...props} title="Copy" icon={<Copy />} />
);
const MoveUpButton = (props: IconButtonProps) => (
  <IconButton {...props} title="Move up" icon={<ArrowUp />} />
);
const MoveDownButton = (props: IconButtonProps) => (
  <IconButton {...props} title="Move down" icon={<ArrowDown />} />
);

function TitleFieldTemplate({ id, title, required, optionalDataControl }: TitleFieldProps) {
  return (
    <div id={id} className="font-semibold">
      {title}
      {required && <span className="text-error"> *</span>}
      {optionalDataControl}
    </div>
  );
}

function DescriptionFieldTemplate({ id, description }: DescriptionFieldProps) {
  if (!description) return null;
  return (
    <div id={id} className="text-sm opacity-70">
      {description as ReactNode}
    </div>
  );
}

function FieldErrorTemplate({ errors = [] }: FieldErrorProps) {
  if (errors.length === 0) return null;
  return (
    <ul className="text-error text-sm">
      {errors.map((e, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: static list
        <li key={i}>{e}</li>
      ))}
    </ul>
  );
}

function FieldHelpTemplate({ help }: FieldHelpProps) {
  if (!help) return null;
  return <div className="text-sm opacity-70">{help as ReactNode}</div>;
}

function ErrorListTemplate({ errors }: ErrorListProps) {
  return (
    <div role="alert" className="alert alert-error mb-4 flex-col items-start">
      {errors.map((e, i) => (
        // biome-ignore lint/suspicious/noArrayIndexKey: static list
        <div key={i}>{e.stack}</div>
      ))}
    </div>
  );
}

function BaseInputTemplate(props: BaseInputTemplateProps) {
  const {
    id,
    htmlName,
    value,
    readonly,
    disabled,
    autofocus,
    onBlur,
    onFocus,
    onChange,
    onChangeOverride,
    options,
    schema,
    type,
    placeholder,
  } = props;
  const inputProps = getInputProps(schema, type, options);
  return (
    <input
      id={id}
      name={htmlName || id}
      className="input input-bordered input-sm w-full"
      readOnly={readonly}
      disabled={disabled}
      autoFocus={autofocus}
      placeholder={placeholder}
      value={value == null ? "" : value}
      {...inputProps}
      onChange={
        onChangeOverride ||
        ((e) => onChange(e.target.value === "" ? options.emptyValue : e.target.value))
      }
      onBlur={(e) => onBlur(id, e.target.value)}
      onFocus={(e) => onFocus(id, e.target.value)}
    />
  );
}

function FieldTemplate(props: FieldTemplateProps) {
  const {
    id,
    label,
    children,
    errors,
    help,
    description,
    hidden,
    required,
    displayLabel,
    schema,
    disabled,
    readonly,
    onKeyRenameBlur,
    onRemoveProperty,
    registry,
    uiSchema,
  } = props;
  if (hidden) return <div className="hidden">{children}</div>;
  const labelEl = displayLabel && label && (
    <label className="font-medium" htmlFor={id}>
      {label}
      {required && <span className="text-error"> *</span>}
    </label>
  );
  const body = (
    <>
      {description}
      {children}
      {errors}
      {help}
    </>
  );
  if (!(ADDITIONAL_PROPERTY_FLAG in schema)) {
    return (
      <div className="flex flex-col gap-1">
        {labelEl}
        {body}
      </div>
    );
  }
  // An entry of an `additionalProperties` dict: editable key + value + remove button.
  const { RemoveButton } = registry.templates.ButtonTemplates;
  const isObject = schema.type === "object";
  const keyInput = (
    <input
      className="input input-bordered input-sm w-full"
      type="text"
      id={`${id}-key`}
      defaultValue={label}
      placeholder="Key"
      onBlur={onKeyRenameBlur}
      key={label}
    />
  );
  const removeButton = (
    <RemoveButton
      id={buttonId(id, "remove")}
      disabled={disabled || readonly}
      onClick={onRemoveProperty}
      uiSchema={uiSchema}
      registry={registry}
    />
  );
  if (isObject) {
    return (
      <div className="border border-base-300 rounded-box p-3 flex flex-col gap-2">
        <div className="flex gap-2 items-center">
          {keyInput}
          {removeButton}
        </div>
        {body}
      </div>
    );
  }
  return (
    <div className="flex gap-2 items-start">
      <div className="flex-1">{keyInput}</div>
      <div className="flex-1 flex flex-col gap-1">{body}</div>
      {removeButton}
    </div>
  );
}

function ObjectFieldTemplate(props: ObjectFieldTemplateProps) {
  const {
    title,
    description,
    properties,
    fieldPathId,
    schema,
    uiSchema,
    formData,
    disabled,
    readonly,
    onAddProperty,
    registry,
    required,
    optionalDataControl,
  } = props;
  const { TitleFieldTemplate, DescriptionFieldTemplate } = registry.templates;
  const { AddButton } = registry.templates.ButtonTemplates;
  const isRoot = fieldPathId.$id === "root";
  return (
    <fieldset
      id={fieldPathId.$id}
      className={isRoot ? "flex flex-col gap-5" : "flex flex-col gap-2"}
    >
      {title && !isRoot && (
        <TitleFieldTemplate
          id={`${fieldPathId.$id}__title`}
          title={title}
          required={required}
          schema={schema}
          uiSchema={uiSchema}
          registry={registry}
          optionalDataControl={optionalDataControl}
        />
      )}
      {description && (
        <DescriptionFieldTemplate
          id={`${fieldPathId.$id}__description`}
          description={description}
          schema={schema}
          uiSchema={uiSchema}
          registry={registry}
        />
      )}
      {properties.map((p) => (
        <div key={p.name}>{p.content}</div>
      ))}
      {canExpand(schema, uiSchema, formData) && (
        <AddButton
          id={buttonId(fieldPathId, "add")}
          onClick={onAddProperty}
          disabled={disabled || readonly}
          uiSchema={uiSchema}
          registry={registry}
        />
      )}
    </fieldset>
  );
}

function ArrayFieldTemplate(props: ArrayFieldTemplateProps) {
  const {
    title,
    schema,
    uiSchema,
    required,
    registry,
    fieldPathId,
    items,
    canAdd,
    disabled,
    readonly,
    onAddClick,
    optionalDataControl,
  } = props;
  const { TitleFieldTemplate, DescriptionFieldTemplate } = registry.templates;
  const { AddButton } = registry.templates.ButtonTemplates;
  return (
    <fieldset id={fieldPathId.$id} className="flex flex-col gap-2">
      {title && (
        <TitleFieldTemplate
          id={`${fieldPathId.$id}__title`}
          title={title}
          required={required}
          schema={schema}
          uiSchema={uiSchema}
          registry={registry}
          optionalDataControl={optionalDataControl}
        />
      )}
      {schema.description && (
        <DescriptionFieldTemplate
          id={`${fieldPathId.$id}__description`}
          description={schema.description}
          schema={schema}
          uiSchema={uiSchema}
          registry={registry}
        />
      )}
      <div className="flex flex-col gap-2">{items}</div>
      {canAdd && (
        <AddButton
          id={buttonId(fieldPathId, "add")}
          onClick={onAddClick}
          disabled={disabled || readonly}
          uiSchema={uiSchema}
          registry={registry}
        />
      )}
    </fieldset>
  );
}

function ArrayFieldItemTemplate(props: ArrayFieldItemTemplateProps) {
  const { children, buttonsProps, hasToolbar } = props;
  const { registry, uiSchema } = props;
  const { CopyButton, MoveDownButton, MoveUpButton, RemoveButton } =
    registry.templates.ButtonTemplates;
  const b = buttonsProps;
  return (
    <div className="flex gap-2 items-start">
      <div className="flex-1">{children}</div>
      {hasToolbar && (
        <div className="flex">
          {(b.hasMoveUp || b.hasMoveDown) && (
            <MoveUpButton
              id={buttonId(b.fieldPathId, "moveUp")}
              disabled={b.disabled || b.readonly || !b.hasMoveUp}
              onClick={b.onMoveUpItem}
              uiSchema={uiSchema}
              registry={registry}
            />
          )}
          {(b.hasMoveUp || b.hasMoveDown) && (
            <MoveDownButton
              id={buttonId(b.fieldPathId, "moveDown")}
              disabled={b.disabled || b.readonly || !b.hasMoveDown}
              onClick={b.onMoveDownItem}
              uiSchema={uiSchema}
              registry={registry}
            />
          )}
          {b.hasCopy && (
            <CopyButton
              id={buttonId(b.fieldPathId, "copy")}
              disabled={b.disabled || b.readonly}
              onClick={b.onCopyItem}
              uiSchema={uiSchema}
              registry={registry}
            />
          )}
          {b.hasRemove && (
            <RemoveButton
              id={buttonId(b.fieldPathId, "remove")}
              disabled={b.disabled || b.readonly}
              onClick={b.onRemoveItem}
              uiSchema={uiSchema}
              registry={registry}
            />
          )}
        </div>
      )}
    </div>
  );
}

export const folderSettingsTemplates: Partial<TemplatesType> = {
  FieldTemplate,
  ObjectFieldTemplate,
  ArrayFieldTemplate,
  ArrayFieldItemTemplate,
  BaseInputTemplate,
  TitleFieldTemplate,
  DescriptionFieldTemplate,
  FieldErrorTemplate,
  FieldHelpTemplate,
  ErrorListTemplate,
  ButtonTemplates: {
    AddButton,
    RemoveButton,
    CopyButton,
    MoveUpButton,
    MoveDownButton,
  } as TemplatesType["ButtonTemplates"],
};
