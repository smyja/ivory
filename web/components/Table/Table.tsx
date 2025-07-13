import cx from 'clsx';
import { useState } from 'react';
import {
  Table,
  TextInput,
  Pagination,
  Checkbox,
  ScrollArea,
  Group,
  Avatar,
  Text,
  rem,
} from '@mantine/core';
import { IconSearch } from '@tabler/icons-react';
import classes from './TableSelection.module.css';

interface DataItem {
  id: string;
  avatar: string;
  name: string;
  email: string;
  job: string;
}
const data: DataItem[] = [
  {
    id: '1',
    avatar:
      'https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-1.png',
    name: 'Robert Wolfkisser',
    job: 'Engineer',
    email: 'rob_wolf@gmail.com',
  },
  {
    id: '2',
    avatar:
      'https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-7.png',
    name: 'Jill Jailbreaker',
    job: 'Engineer',
    email: 'jj@breaker.com',
  },
  {
    id: '3',
    avatar:
      'https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-2.png',
    name: 'Henry Silkeater',
    job: 'Designer',
    email: 'henry@silkeater.io',
  },
  {
    id: '4',
    avatar:
      'https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-3.png',
    name: 'Bill Horsefighter',
    job: 'Designer',
    email: 'bhorsefighter@gmail.com',
  },
  {
    id: '5',
    avatar:
      'https://raw.githubusercontent.com/mantinedev/mantine/master/.demo/avatars/avatar-10.png',
    name: 'Jeremy Footviewer',
    job: 'Manager',
    email: 'jeremy@foot.dev',
  },
];

export function TableSelection() {
  const [selection, setSelection] = useState(['1']);
  const [searchQuery, setSearchQuery] = useState('');
  const [activePage, setPage] = useState(1);
  const itemsPerPage = 2; // Number of items per page

  const toggleRow = (id: string) =>
    setSelection((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id]
    );
  const toggleAll = () =>
    setSelection((current) => (current.length === data.length ? [] : data.map((item) => item.id)));
  const filterData = (data: DataItem[], query: string): DataItem[] => {
    const lowerCaseQuery = query.toLowerCase();
    return data.filter(
      (item) =>
        item.name.toLowerCase().includes(lowerCaseQuery) ||
        item.email.toLowerCase().includes(lowerCaseQuery) ||
        item.job.toLowerCase().includes(lowerCaseQuery)
    );
  };

  const filteredData = filterData(data, searchQuery);

  const totalPages = Math.ceil(filteredData.length / itemsPerPage);
  const paginatedData = filteredData.slice(
    (activePage - 1) * itemsPerPage,
    activePage * itemsPerPage
  );

  // Generate rows for the current page
  const rows = paginatedData.map((item) => {
    const selected = selection.includes(item.id);
    return (
      <Table.Tr key={item.id} className={cx({ [classes.rowSelected]: selected })}>
        <Table.Td>
          <Checkbox checked={selected} onChange={() => toggleRow(item.id)} />
        </Table.Td>
        <Table.Td>
          <Group gap="sm">
            <Avatar size={26} src={item.avatar} radius={26} />
            <Text size="sm" fw={500}>
              {item.name}
            </Text>
          </Group>
        </Table.Td>
        <Table.Td>{item.email}</Table.Td>
        <Table.Td>{item.job}</Table.Td>
      </Table.Tr>
    );
  });

  return (
    <>
      <TextInput
        placeholder="Search..."
        leftSection={<IconSearch style={{ width: rem(16), height: rem(16) }} stroke={1.5} />}
        value={searchQuery}
        onChange={(event) => {
          setSearchQuery(event.currentTarget.value);
          setPage(1); // Reset to first page on search
        }}
        mb="md"
      />
      <Text size="sm" c="dimmed" mb="md">
        {selection.length} item(s) selected
      </Text>
      <ScrollArea>
        <Table miw={800} verticalSpacing="sm">
          <Table.Thead>
            <Table.Tr>
              <Table.Th style={{ width: rem(40) }}>
                <Checkbox
                  onChange={toggleAll}
                  checked={selection.length === data.length}
                  indeterminate={selection.length > 0 && selection.length !== data.length}
                />
              </Table.Th>
              <Table.Th>User</Table.Th>
              <Table.Th>Email</Table.Th>
              <Table.Th>Job</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>{rows}</Table.Tbody>
        </Table>
      </ScrollArea>
      <Pagination mt={10} value={activePage} onChange={setPage} total={totalPages} />
    </>
  );
}
