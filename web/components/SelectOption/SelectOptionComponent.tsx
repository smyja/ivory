import { useState } from 'react';
import { Combobox, Group, Input, InputBase, Text, useCombobox } from '@mantine/core';

interface Item {
  emoji: string;
  value: string;
  description: string;
}


const dataFormats: Item[] = [
    { emoji: '🤗', value: 'HuggingFace', description: 'AI model repository and collaboration platform'},
    { emoji: '📊', value: 'CSV', description: 'CSV file format for tabular data'},
    { emoji: '{ }', value: 'JSON', description: 'Json, a lightweight data interchange format'},
    { emoji: '📦', value: 'Parquet', description: 'Columnar storage file format'},
  ];
function SelectOption({ emoji, value, description }: Item) {
  return (
    <Group>
      <Text fz={20}>{emoji}</Text>
      <div>
        <Text fz="sm" fw={500}>
          {value}
        </Text>
        <Text fz="xs" opacity={0.6}>
          {description}
        </Text>
      </div>
    </Group>
  );
}

export function SelectOptionComponent() {
  const combobox = useCombobox({
    onDropdownClose: () => combobox.resetSelectedOption(),
  });

  const [value, setValue] = useState<string | null>(null);
  const selectedOption = dataFormats.find((item) => item.value === value);

  const options = dataFormats.map((item) => (
    <Combobox.Option value={item.value} key={item.value}>
      <SelectOption {...item} />
    </Combobox.Option>
  ));

  return (
    <Combobox
      store={combobox}
      withinPortal={false}
      onOptionSubmit={(val) => {
        setValue(val);
        combobox.closeDropdown();
      }}
    >
      <Combobox.Target>
        <InputBase
          component="button"
          type="button"
          pointer
          rightSection={<Combobox.Chevron />}
          onClick={() => combobox.toggleDropdown()}
          rightSectionPointerEvents="none"
          multiline
        >
          {selectedOption ? (
            <SelectOption {...selectedOption} />
          ) : (
            <Input.Placeholder>Select Data source</Input.Placeholder>
          )}
        </InputBase>
      </Combobox.Target>

      <Combobox.Dropdown>
        <Combobox.Options>{options}</Combobox.Options>
      </Combobox.Dropdown>
    </Combobox>
  );
}