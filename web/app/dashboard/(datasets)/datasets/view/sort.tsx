import { useState, useMemo } from 'react';
import { Button, Combobox, ActionIcon, Tooltip, useCombobox } from '@mantine/core';
import { IconSortAscending, IconSortDescending } from '@tabler/icons-react';

type Direction = 'asc' | 'desc';

interface SortButtonProps {
  columns?: string[];
  onSort?: (column: string, direction: Direction) => void;
  initialColumn?: string | null;
  initialDirection?: Direction;
}

export function SortButton({
  columns,
  onSort,
  initialColumn = null,
  initialDirection = 'asc',
}: SortButtonProps) {
  const [search, setSearch] = useState('');
  const [selectedItem, setSelectedItem] = useState<string | null>(initialColumn);
  const [direction, setDirection] = useState<Direction>(initialDirection);

  const opts = useMemo(() => {
    const source = (columns && columns.length ? columns : ['id', 'text', 'date']).filter(
      (c) => !!c && typeof c === 'string'
    );
    return source
      .filter((item) => item.toLowerCase().includes(search.toLowerCase().trim()))
      .map((item) => (
        <Combobox.Option value={item} key={item}>
          {item}
        </Combobox.Option>
      ));
  }, [columns, search]);

  const combobox = useCombobox({
    onDropdownClose: () => {
      combobox.resetSelectedOption();
      combobox.focusTarget();
      setSearch('');
    },
    onDropdownOpen: () => {
      combobox.focusSearchInput();
    },
  });

  return (
    <Combobox
      store={combobox}
      width={240}
      position="bottom-start"
      withArrow
      withinPortal={false}
      onOptionSubmit={(val) => {
        setSelectedItem(val);
        combobox.closeDropdown();
        onSort && onSort(val, direction);
      }}
    >
      <Combobox.Target withAriaAttributes={false}>
        <Button
          variant="light"
          onClick={() => combobox.toggleDropdown()}
          rightSection={
            <Tooltip label={`Direction: ${direction.toUpperCase()}`} withinPortal>
              <ActionIcon
                size="sm"
                variant="subtle"
                color="gray"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                  e.stopPropagation();
                  const next: Direction = direction === 'asc' ? 'desc' : 'asc';
                  setDirection(next);
                  if (selectedItem && onSort) onSort(selectedItem, next);
                }}
              >
                {direction === 'asc' ? (
                  <IconSortAscending size="1rem" />
                ) : (
                  <IconSortDescending size="1rem" />
                )}
              </ActionIcon>
            </Tooltip>
          }
        >
          {selectedItem ? `Sort: ${selectedItem}` : 'Sort By'}
        </Button>
      </Combobox.Target>

      <Combobox.Dropdown>
        <Combobox.Search
          value={search}
          onChange={(event) => setSearch(event.currentTarget.value)}
          placeholder="Search columns"
        />
        <Combobox.Options>
          {opts.length > 0 ? opts : <Combobox.Empty>Nothing found</Combobox.Empty>}
        </Combobox.Options>
      </Combobox.Dropdown>
    </Combobox>
  );
}
