// src/components/CustomPagination.tsx
import React from 'react';
import { ActionIcon, Group, Text } from '@mantine/core';
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react';

interface CustomPaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  disabled?: boolean;
}

const CustomPagination: React.FC<CustomPaginationProps> = ({
  currentPage,
  totalPages,
  onPageChange,
  disabled = false,
}) => (
  <Group gap={5} align="center">
    <ActionIcon
      color="gray"
      onClick={() => onPageChange(Math.max(1, currentPage - 1))}
      disabled={disabled || currentPage === 1}
    >
      <IconChevronLeft size="1rem" />
    </ActionIcon>
    <Text size="sm" style={{ opacity: disabled ? 0.7 : 1 }}>
      {currentPage} of {totalPages.toLocaleString()}
    </Text>
    <ActionIcon
      color="gray"
      onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
      disabled={disabled || currentPage === totalPages}
    >
      <IconChevronRight size="1rem" />
    </ActionIcon>
  </Group>
);

export default CustomPagination;
