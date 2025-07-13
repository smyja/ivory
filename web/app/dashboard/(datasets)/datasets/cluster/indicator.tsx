import React from 'react';
import { Badge, Indicator } from '@mantine/core';

interface IndicatorBadgeProps {
  color: string;
  label: string;
  processing?: boolean;
  textColor?: string;
}

const IndicatorBadge: React.FC<IndicatorBadgeProps> = ({
  color,
  label,
  processing = false,
  textColor = color,
}) => (
  <Indicator
    inline
    size={10}
    offset={7}
    position="middle-start"
    color={color}
    processing={processing}
    withBorder
  >
    <Badge
      variant="outline"
      styles={(theme) => ({
        root: {
          backgroundColor: 'transparent',
          border: 'none',
          paddingLeft: theme.spacing.xl,
          color: textColor, // Use the passed textColor here
          fontWeight: 400, // Set font weight
          textTransform: 'none', // Remove uppercase transformation
          fontSize: '14px',
        },
      })}
    >
      {label}
    </Badge>
  </Indicator>
);

export default IndicatorBadge;
