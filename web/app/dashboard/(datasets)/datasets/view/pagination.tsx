// src/components/CustomPagination.tsx
import React from 'react';
import { ActionIcon, Group, Text } from '@mantine/core';
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react';

interface CustomPaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const CustomPagination: React.FC<CustomPaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
}) => {
  return (
    <Group gap={5} align="center">
      <ActionIcon
        color="gray"
        onClick={() => onPageChange(Math.max(1, currentPage - 1))}
        disabled={currentPage === 1}
      >
        <IconChevronLeft size="1rem" />
      </ActionIcon>
      <Text size="sm">
        {currentPage} of {totalPages.toLocaleString()}
      </Text>
      <ActionIcon
        color="gray"
        onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
        disabled={currentPage === totalPages}
      >
        <IconChevronRight size="1rem" />
      </ActionIcon>
    </Group>
  );
};

export default CustomPagination;