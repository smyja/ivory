import { useState, useEffect } from 'react';
import { Box, UnstyledButton, Collapse, Group, rem } from '@mantine/core';
import { usePathname, useRouter } from 'next/navigation';
import {
  IconHome,
  IconBrain,
  IconSettings,
  IconPlugConnected,
  IconMessageChatbot,
  IconFiles,
  IconChevronRight,
  IconSwitchHorizontal,
  IconLogout,
  IconDatabase,
} from '@tabler/icons-react';
import Link from 'next/link';
import classes from './NavbarSimpleColored.module.css';

interface NavItem {
  link?: string;
  label: string;
  icon?: any;
  links?: { link: string; label: string }[];
}

const data: NavItem[] = [
  { link: '/dashboard', label: 'Home', icon: IconHome },
  { link: '/dashboard/chat', label: 'Chat', icon: IconMessageChatbot },
  { link: '/dashboard/connectors', label: 'Connectors', icon: IconPlugConnected },
  { link: '/dashboard/documents', label: 'Documents', icon: IconFiles },
  {
    label: 'Data & clusters',
    icon: IconDatabase,
    links: [
      { link: '/dashboard/datasets', label: 'Datasets' },
      { link: '/dashboard/datasets/create', label: 'Create Dataset' },
      { link: '/dashboard/datasets/cluster/history', label: 'Clustering History' },
    ],
  },
  { link: '/dashboard/insight', label: 'Insight', icon: IconBrain },
  { link: '/dashboard/settings', label: 'Settings', icon: IconSettings },
];

export function NavbarSimpleColored() {
  const pathname = usePathname();
  const router = useRouter();

  const findLabelByPath = (path: string) => {
    const matchingItem = data.find((item) =>
      item.link ? item.link === path : item.links?.some((subLink) => subLink.link === path)
    );
    return matchingItem ? matchingItem.label : 'Home';
  };

  const [active, setActive] = useState(findLabelByPath(pathname));
  const [openGroup, setOpenGroup] = useState('');

  useEffect(() => {
    setActive(findLabelByPath(pathname));
  }, [pathname]);

  const handleLinkClick = (item: NavItem) => {
    setActive(item.label);
    if (item.link) {
      router.push(item.link);
    }
  };

  return (
    <>
      <div className={classes.navbarMain}>
        {data.map((item) => {
          if (item.links) {
            return (
              <Box key={item.label} className={classes.collapsibleLink}>
                <UnstyledButton
                  onClick={() => setOpenGroup((prev) => (prev === item.label ? '' : item.label))}
                  className={`${classes.control}`}
                >
                  <Group justify="space-between">
                    <Box style={{ display: 'flex', alignItems: 'center' }}>
                      <item.icon className={classes.linkIcon} stroke={1.5} />
                      <span>{item.label}</span>
                    </Box>
                    <IconChevronRight
                      className={classes.chevron}
                      stroke={1.5}
                      style={{
                        transform: openGroup === item.label ? 'rotate(90deg)' : 'none',
                        marginLeft: rem(-5),
                      }}
                    />
                  </Group>
                </UnstyledButton>

                <Collapse in={openGroup === item.label}>
                  <div className={classes.subLinks}>
                    {item.links.map((subLink) => (
                      <Link
                        href={subLink.link}
                        key={subLink.label}
                        onClick={() => handleLinkClick(subLink)}
                      >
                        <div
                          className={`${classes.link} ${
                            pathname === subLink.link ? classes.active : ''
                          }`}
                        >
                          {subLink.label}
                        </div>
                      </Link>
                    ))}
                  </div>
                </Collapse>
              </Box>
            );
          }
          return item.link ? (
            <Link href={item.link} key={item.label} onClick={() => handleLinkClick(item)}>
              <div className={`${classes.link} ${active === item.label ? classes.active : ''}`}>
                <item.icon className={classes.linkIcon} stroke={1.5} />
                <span>{item.label}</span>
              </div>
            </Link>
          ) : null;
        })}
      </div>

      <div className={classes.footer}>
        <Link href="/dashboard/settings" className={classes.link}>
          <IconSwitchHorizontal className={classes.linkIcon} stroke={1.5} />
          <span>Change account</span>
        </Link>

        <a href="#" className={classes.link} onClick={(event) => event.preventDefault()}>
          <IconLogout className={classes.linkIcon} stroke={1.5} />
          <span>Logout</span>
        </a>
      </div>
    </>
  );
}
